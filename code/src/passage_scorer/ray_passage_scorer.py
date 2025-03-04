import json
import os
import pyterrier as pt
from tqdm import tqdm
import gzip
from trectools import TrecQrel, TrecRun, TrecEval
from chatnoir_pyterrier import ChatNoirRetrieve
from requests.exceptions import ReadTimeout
import ray

ray.init()


@ray.remote
def ray_wrapper(job_id, num_jobs, qrels_cache, docs_to_score):
    def load_config(filename="/mnt/ceph/storage/data-tmp/current/ho62zoq/thesis-fhofer/code/src/config.json"):
        with open(filename, "r") as f:
            config = json.load(f)
        return config

    # Get the configuration settings
    config = load_config()

    PT_RETRIEVERS = config['PT_RETRIEVERS']

    DOCUMENT_DATASET_SOURCE_NAME = config['DOCUMENT_DATASET_SOURCE_NAME']
    DOCUMENT_DATASET_SOURCE_NAME_PYTERRIER = config['DOCUMENT_DATASET_SOURCE_NAME_PYTERRIER']

    SOURCE_PATH = os.path.join(config['DATA_PATH'], DOCUMENT_DATASET_SOURCE_NAME)

    DOCUMENT_DATASET_SOURCE_INDEX_PATH = os.path.join(SOURCE_PATH, config['DOCUMENT_DATASET_SOURCE_INDEX_PATH'])

    PASSAGE_DATASET_SOURCE_PATH = os.path.join(SOURCE_PATH, config['PASSAGE_DATASET_SOURCE_PATH'])

    PASSAGE_DATASET_SOURCE_SCORE_AQ_PATH = os.path.join(SOURCE_PATH, config['PASSAGE_DATASET_SOURCE_SCORE_AQ_PATH'])
    PASSAGE_DATASET_SOURCE_SCORE_REL_PATH = os.path.join(SOURCE_PATH, config['PASSAGE_DATASET_SOURCE_SCORE_REL_PATH'])

    PASSAGE_ID_SEPARATOR = config['PASSAGE_ID_SEPARATOR']

    CHATNOIR_RETRIEVAL = config['CHATNOIR_RETRIEVAL']
    CHATNOIR_SOURCE_INDICES = config['CHATNOIR_SOURCE_INDICES']
    CHATNOIR_API_KEY = config['CHATNOIR_API_KEY']

    JOB_ID = job_id
    NUM_JOBS = num_jobs
    QIDS = None

    # Distribute QIDs among jobs
    keys = list(qrels_cache.keys())
    keys = [key for key in keys if not os.path.exists(os.path.join(PASSAGE_DATASET_SOURCE_SCORE_REL_PATH, f"qid_{key}.jsonl.gz"))]

    total_qids = len(keys)

    # Determine the range of QIDs for this job
    qids_per_job = (total_qids + NUM_JOBS - 1) // NUM_JOBS  # Ceiling division
    start_index = (JOB_ID - 1) * qids_per_job
    end_index = min(start_index + qids_per_job, total_qids)

    # Assign QIDs for the current job
    QIDS = keys[start_index:end_index]

    # Initialize PyTerrier and Tokenizer
    if not pt.java.started():
        pt.java.init()
    tokeniser = pt.java.autoclass('org.terrier.indexing.tokenisation.Tokeniser').getTokeniser()

    # Tokenize text

    def pt_tokenize(text):
        return ' '.join(tokeniser.getTokens(text))

    # Read passages and cache them
    passages_cache = {}
    with gzip.open(PASSAGE_DATASET_SOURCE_PATH, 'rt', encoding='UTF-8') as file:
        for line in tqdm(file, desc='Caching passages', unit='passage'):
            line = json.loads(line)
            docno, passageno = line['docno'].split(PASSAGE_ID_SEPARATOR)
            if docno not in passages_cache:
                passages_cache[docno] = []
            passages_cache[docno] += [line]

    if CHATNOIR_RETRIEVAL:
        PT_RETRIEVERS = ['BM25_chatnoir']
        retrievers = {}
        for retriever in PT_RETRIEVERS:
            retrievers[retriever] = ChatNoirRetrieve(api_key=CHATNOIR_API_KEY,
                                                     features=[],
                                                     index=CHATNOIR_SOURCE_INDICES,
                                                     search_method="bm25",
                                                     num_results=100)
    else:
        index_ref = pt.IndexRef.of(DOCUMENT_DATASET_SOURCE_INDEX_PATH + '/data.properties')
        dataset_index = pt.IndexFactory.of(index_ref)

        # retrieval models
        retrievers = {}
        for retriever in PT_RETRIEVERS:
            retrievers[retriever] = pt.terrier.Retriever(dataset_index, wmodel=retriever)

    # Get reciprocal rank of the original document in a run

    def get_reciprocal_rank_of_docno(run_data, docno):
        for index, row in run_data.iterrows():
            if row['docid'] == docno:
                return 1 / (index + 1)
        return 0

    # Infer run from passage text and retrieve top 10 passages

    def get_infered_run(retriever, passage_text, system_name, docno):
        run = TrecRun()
        run_wod = TrecRun()

        # Retrieve the top 11 entries
        if CHATNOIR_RETRIEVAL:
            try:
                run.run_data = retriever.search(passage_text).loc[
                    :, ['qid', 'docno', 'rank', 'score']].rename(
                    columns={'qid': 'query', 'docno': 'docid', 'score': 'score'}).head(11)
            except ReadTimeout as e:
                print(f"{e} for passage of {docno}: \n {passage_text}")
                return None, None, None

            except RuntimeError as e:
                print(f"{e} for passage of {docno}: \n {passage_text}")
                return None, None, None
            except Exception as e:
                print(f"Unknown error {e} for passage of {docno}: \n {passage_text}")
                return

        else:
            run.run_data = retriever.search(pt_tokenize(passage_text)).loc[
                :, ['qid', 'docno', 'rank', 'score']].rename(
                columns={'qid': 'query', 'docno': 'docid', 'score': 'score'}).head(11)

        run.run_data['query'] = 0  # Dummy value to enable merge of run and qrels (TrecEval)
        run.run_data['q0'] = 'Q0'  # Dummy value to get ndcg score (TrecEval)
        run.run_data['system'] = system_name  # Dummy value to get ndcg score (TrecEval)

        # Drop the last row to keep top 10, can contain orginal document
        run.run_data = run.run_data.iloc[:-1]

        reciprocal_rank_docno = get_reciprocal_rank_of_docno(run.run_data, docno)

        # If docno is in top 10, remove it; otherwise, remove the last entry
        if docno in run.run_data['docid'].values:
            run_wod.run_data = run.run_data[run.run_data['docid'] != docno]
        else:
            run_wod.run_data = run.run_data

        return run, run_wod, reciprocal_rank_docno

    # Get all qrels for a query and remove original document if specified
    def get_qrels_for_query(qid, docno, include_original_document):
        qrels_for_query = TrecQrel()
        qrels_for_query.qrels_data = qrels_cache[qid]
        # Remove original document if specified
        if not include_original_document:
            qrels_for_query.qrels_data = qrels_for_query.qrels_data[qrels_for_query.qrels_data['docid'] != docno]
        return qrels_for_query

    # Evaluate run using TrecEval
    def evaluate_run(run, qrels_for_query):
        te = TrecEval(run, qrels_for_query)
        p10_score = float(te.get_precision(depth=10, removeUnjudged=True))
        ndcg10_score = float(te.get_ndcg(depth=10, removeUnjudged=True))
        return p10_score, ndcg10_score

    # Process a single QID
    def process_qid(qid, docs_to_score):

        scored_docs_count = 0
        results = []
        relevant_results = []

        print(f"Processing QID {qid} in process with Job: {JOB_ID}")
        for docno in qrels_cache[qid]['docid']:
            label = int(qrels.loc[(qrels['qid'] == qid) & (qrels['docno'] == docno)]['label'].iloc[0])
            # All labels smaller equal 0 are considered as 0 (non-relevant)
            if label < 0:
                label = 0
            # Check if docno should be scored for this qid
            if docno in docs_to_score[qid][str(label)]:
                # Check if docno should be scored
                if docno in passages_cache:  # Should never be false if docno is in docs_to_score
                    scored_docs_count += 1
                    for passage in passages_cache[docno]:
                        # wod = without original document
                        qrels_for_query = get_qrels_for_query(qid, docno, include_original_document=True)
                        qrels_for_query_wod = get_qrels_for_query(qid, docno, include_original_document=False)

                        # Infer runs for all retrievers
                        runs = {}
                        runs_wod = {}
                        reciprocal_ranks = {}

                        for retriever in PT_RETRIEVERS:
                            runs[retriever], runs_wod[retriever], reciprocal_ranks[retriever] = get_infered_run(
                                retrievers[retriever], passage['text'], retriever, docno)

                        # Evaluate passage scores
                        p10 = {}
                        ndcg10 = {}

                        for retriever in PT_RETRIEVERS:
                            if runs[retriever] is None:
                                continue
                            p10[retriever], ndcg10[retriever] = evaluate_run(runs[retriever], qrels_for_query)
                            p10[retriever + '_wod'], ndcg10[retriever + '_wod'] = evaluate_run(runs_wod[retriever],
                                                                                               qrels_for_query_wod)

                        scores = {'qid': qid, 'docno': passage['docno'], 'label': label}
                        for retriever in PT_RETRIEVERS:
                            if runs[retriever] is None:
                                scores['p10_' + retriever] = 0.0
                                scores['p10_wod_' + retriever] = 0.0
                                scores['ndcg10_' + retriever] = 0.0
                                scores['ndcg10_wod_' + retriever] = 0.0
                                scores['reciprocal_rank_docno_' + retriever] = 0.0
                            else:
                                scores['p10_' + retriever] = p10[retriever]
                                scores['p10_wod_' + retriever] = p10[retriever + '_wod']
                                scores['ndcg10_' + retriever] = ndcg10[retriever]
                                scores['ndcg10_wod_' + retriever] = ndcg10[retriever + '_wod']
                                scores['reciprocal_rank_docno_' + retriever] = reciprocal_ranks[retriever]

                        results.append(scores)

                        if qrels.loc[(qrels['qid'] == qid) & (qrels['docno'] == docno)]['label'].iloc[0] > 0:
                            relevant_results.append(scores)

        return results, relevant_results, scored_docs_count

    for QID in QIDS:
        relevant_path = os.path.join(PASSAGE_DATASET_SOURCE_SCORE_REL_PATH, f"qid_{QID}.jsonl.gz")
        all_path = os.path.join(PASSAGE_DATASET_SOURCE_SCORE_AQ_PATH, f"qid_{QID}.jsonl.gz")

        # Check if scores already exist
        if os.path.exists(relevant_path) and os.path.exists(all_path):
            print(f"Scores for QID {QID} already exist. Exiting.")
            continue

        results, relevant_results, docs_count = process_qid(QID, docs_to_score)

        with gzip.open(relevant_path, 'wt', encoding='UTF-8') as relevant_qrels_file:
            for scores in relevant_results:
                relevant_qrels_file.write(json.dumps(scores) + '\n')

        with gzip.open(all_path, 'wt', encoding='UTF-8') as all_qrels_file:
            for scores in results:
                all_qrels_file.write(json.dumps(scores) + '\n')

        print(f"Job {job_id} processed and saved {docs_count} documents for QID {QID}")


"""
Get list of doc ids that should be chunked
For each QID, chunk 50 non relevant documents with a label <= 0
For each QID, chunk 50 relevant documents for each label > 0
"""
def get_docs_to_chunk(dataset):
    dict = {}

    for qrel in dataset.irds_ref().qrels_iter():
        qid = qrel.query_id
        doc_id = qrel.doc_id
        label = qrel.relevance

        if qid not in dict:
            dict[qid] = {}

        # Map non-relevant documents to label 0
        if label <= 0:
            if '0' not in dict[qid]:
                dict[qid]['0'] = []
            dict[qid]['0'] += [doc_id]

        if label > 0:
            lable_str = str(label)
            if lable_str not in dict[qid]:
                dict[qid][lable_str] = []
            dict[qid][lable_str] += [doc_id]

    # Round to smallest label count or 50
    for qid in dict:
        for label in dict[qid]:
            dict[qid][label] = dict[qid][label][:50]

    return dict


if __name__ == '__main__':

    with open("/mnt/ceph/storage/data-tmp/current/ho62zoq/thesis-fhofer/code/src/config.json", "r") as f:
        config = json.load(f)
    DOCUMENT_DATASET_SOURCE_NAME_PYTERRIER = config['DOCUMENT_DATASET_SOURCE_NAME_PYTERRIER']
    print(DOCUMENT_DATASET_SOURCE_NAME_PYTERRIER)

    # Read qrels and cache relevant qrels
    dataset = pt.get_dataset(DOCUMENT_DATASET_SOURCE_NAME_PYTERRIER)
    qrels = dataset.get_qrels(variant='relevance')
    qrels_cache = {}
    for index, row in tqdm(qrels.iterrows(), desc='Caching qrels', unit='qrel'):
        if row['qid'] not in qrels_cache:
            qrels_cache[row['qid']] = qrels.loc[
                (qrels['qid'] == row['qid'])
            ].rename(columns={'qid': 'query', 'docno': 'docid', 'label': 'rel'})  # Rename columns
            qrels_cache[row['qid']]['query'] = 0  # Dummy value to enable merge of run and qrels (TrecEval)
    
    # Avoids chunking docs for qid_x that have not been selected in document_chunker_serial.py
    # Elsewise, if another qid_y got an document assigned that has a qrel with qid_x, the document would be scored
    docs_to_score = get_docs_to_chunk(dataset)

    NUM_WORKERS = 20

    futures = []
    for job_id in range(1, NUM_WORKERS + 1):
        futures.append(ray_wrapper.remote(job_id, NUM_WORKERS, qrels_cache, docs_to_score))

    # Wait for all tasks to complete
    ray.get(futures)
