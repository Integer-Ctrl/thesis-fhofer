import json
import os
import pyterrier as pt
import gzip
from trectools import TrecQrel, TrecRun, TrecEval
from concurrent.futures import ProcessPoolExecutor
import threading
import time

# Load the configuration settings
pwd = os.path.dirname(os.path.abspath(__file__))


def load_config(filename=pwd + "/../config.json"):
    with open(filename, "r") as f:
        config = json.load(f)
    return config


# Get the configuration settings
config = load_config()

PT_RETRIEVERS = config['PT_RETRIEVERS']

DOCUMENT_DATASET_OLD_NAME = config['DOCUMENT_DATASET_OLD_NAME']
DOCUMENT_DATASET_OLD_NAME_PYTERRIER = config['DOCUMENT_DATASET_OLD_NAME_PYTERRIER']

OLD_PATH = os.path.join(config['DATA_PATH'], DOCUMENT_DATASET_OLD_NAME)

DOCUMENT_DATASET_OLD_INDEX_PATH = os.path.join(OLD_PATH, config['DOCUMENT_DATASET_OLD_INDEX_PATH'])

PASSAGE_DATASET_OLD_PATH = os.path.join(OLD_PATH, config['PASSAGE_DATASET_OLD_PATH'])

PASSAGE_DATASET_OLD_SCORE_AQ_PATH = os.path.join(OLD_PATH, config['PASSAGE_DATASET_OLD_SCORE_AQ_PATH'])
PASSAGE_DATASET_OLD_SCORE_REL_PATH = os.path.join(OLD_PATH, config['PASSAGE_DATASET_OLD_SCORE_REL_PATH'])

PASSAGE_ID_SEPARATOR = config['PASSAGE_ID_SEPARATOR']

# Create lock for writing to files
WRITE_LOCK = threading.Lock()


# Document yield function for indexing without duplicates
def yield_docs(dataset):
    known_docnos = set()
    for i in dataset.irds_ref().docs_iter():
        if i.doc_id not in known_docnos:
            known_docnos.add(i.doc_id)
            yield {'docno': i.doc_id, 'text': i.default_text()}


# Index dataset
dataset = pt.get_dataset(DOCUMENT_DATASET_OLD_NAME_PYTERRIER)
if not os.path.exists(DOCUMENT_DATASET_OLD_INDEX_PATH):
    indexer = pt.IterDictIndexer(DOCUMENT_DATASET_OLD_INDEX_PATH)
    index_ref = indexer.index(yield_docs(dataset),
                              meta={'docno': 50, 'text': 20000})

# Read passages and cache them
passages_cache = {}
passage_counter = 0
with gzip.open(PASSAGE_DATASET_OLD_PATH, 'rt', encoding='UTF-8') as file:
    for line in file:
        line = json.loads(line)
        docno, passageno = line['docno'].split(PASSAGE_ID_SEPARATOR)
        if docno not in passages_cache:
            passages_cache[docno] = []
        passages_cache[docno] += [line]
        passage_counter += 1
print(f"Loaded {passage_counter} passages.")

# Read qrels and cache relevant qrels
qrels = dataset.get_qrels(variant='relevance')
qrels_cache = {}
for index, row in qrels.iterrows():
    if row['qid'] not in qrels_cache:
        qrels_cache[row['qid']] = qrels.loc[
            (qrels['qid'] == row['qid'])
        ].rename(columns={'qid': 'query', 'docno': 'docid', 'label': 'rel'})  # Rename columns
        qrels_cache[row['qid']]['query'] = 0  # Dummy value to enable merge of run and qrels (TrecEval)


# Get reciprocal rank of the original document in a run
def get_reciprocal_rank_of_docno(run_data, docno):
    for index, row in run_data.iterrows():
        if row['docid'] == docno:
            return 1 / (index + 1)
    return 0


# Infer run from passage text and retrieve top 10 passages
def get_infered_run(retriever, tokenized_passage_text, system_name, docno):
    run = TrecRun()
    run_wod = TrecRun()

    # Retrieve the top 11 entries
    run.run_data = retriever.search(tokenized_passage_text).loc[
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
def get_qrels_for_query(qid, include_original_document):
    qrels_for_query = TrecQrel()
    qrels_for_query.qrels_data = qrels_cache[qid]
    # Remove original document if specified
    if not include_original_document:
        qrels_for_query.qrels_data = qrels_for_query.qrels_data[qrels_for_query.qrels_data['docid'] != qid]
    return qrels_for_query


# Evaluate run using TrecEval
def evaluate_run(run, qrels_for_query):
    te = TrecEval(run, qrels_for_query)
    p10_score = float(te.get_precision(depth=10, removeUnjudged=True))
    ndcg10_score = float(te.get_ndcg(depth=10, removeUnjudged=True))
    return p10_score, ndcg10_score


# # retrieval models
# retrievers = {}
# for retriever in PT_RETRIEVERS:
#     retrievers[retriever] = pt.terrier.Retriever(dataset_index, wmodel=retriever)


def process_qid(args):
    qids, index_path = args

    if not pt.java.started():
        pt.java.init()
    tokeniser = pt.java.autoclass('org.terrier.indexing.tokenisation.Tokeniser').getTokeniser()

    # Tokenize text
    def pt_tokenize(text):
        return ' '.join(tokeniser.getTokens(text))

    index_ref = pt.IndexRef.of(index_path + '/data.properties')
    dataset_index = pt.IndexFactory.of(index_ref)

    retrievers = {
        retriever: pt.terrier.Retriever(dataset_index, wmodel=retriever) for retriever in PT_RETRIEVERS
    }

    results = []
    relevant_results = []

    pid = os.getpid()

    for qid in qids:
        print(f"Processing QID {qid} in process with PID: {pid}")
        for docno in qrels_cache[qid]['docid']:
            for passage in passages_cache[docno]:

                qrels_for_query = get_qrels_for_query(qid, include_original_document=True)
                qrels_for_query_wod = get_qrels_for_query(qid, include_original_document=False)

                runs = {}
                runs_wod = {}
                reciprocal_ranks = {}

                # Use local retrievers for each process
                for retriever in PT_RETRIEVERS:
                    runs[retriever], runs_wod[retriever], reciprocal_ranks[retriever] = get_infered_run(
                        retrievers[retriever], pt_tokenize(passage['text']), retriever, docno)
                p10 = {}
                ndcg10 = {}

                for retriever in PT_RETRIEVERS:
                    p10[retriever], ndcg10[retriever] = evaluate_run(runs[retriever], qrels_for_query)
                    p10[retriever + '_wod'], ndcg10[retriever + '_wod'] = evaluate_run(runs_wod[retriever],
                                                                                       qrels_for_query_wod)

                scores = {'qid': qid, 'docno': passage['docno']}
                for retriever in PT_RETRIEVERS:
                    scores['p10_' + retriever] = p10[retriever]
                    scores['p10_wod_' + retriever] = p10[retriever + '_wod']
                    scores['ndcg10_' + retriever] = ndcg10[retriever]
                    scores['ndcg10_wod_' + retriever] = ndcg10[retriever + '_wod']
                    scores['reciprocal_rank_docno_' + retriever] = reciprocal_ranks[retriever]

                results.append(scores)

                if qrels.loc[(qrels['qid'] == qid) & (qrels['docno'] == docno)]['label'].iloc[0] > 0:
                    relevant_results.append(scores)

        with WRITE_LOCK:
            print(f"Writing results for QID {qid} to file in process with PID: {pid}")
            with gzip.open(PASSAGE_DATASET_OLD_SCORE_REL_PATH, 'at', encoding='UTF-8') as relevant_qrels_file, \
                    gzip.open(PASSAGE_DATASET_OLD_SCORE_AQ_PATH, 'at', encoding='UTF-8') as all_qrels_file:

                # Write to all QRELs file
                for scores in results:
                    all_qrels_file.write(json.dumps(scores) + '\n')

                # Write to relevant QRELs file only if relevance label > 0
                for scores in relevant_results:
                    relevant_qrels_file.write(json.dumps(scores) + '\n')

    print(f"Finished processing QIDs in process with PID: {pid}")


# Parallel processing
if __name__ == "__main__":
    NUM_WORKERS = 32
    start_time = time.time()

    # Clear files before writing in parallel
    with open(PASSAGE_DATASET_OLD_SCORE_REL_PATH, 'wt') as relevant_qrels_file, \
            open(PASSAGE_DATASET_OLD_SCORE_AQ_PATH, 'wt') as all_qrels_file:
        pass

    # Calculate chunk size to evenly distribute keys
    qids = list(qrels_cache.keys())
    chunk_size = (len(qids) + NUM_WORKERS - 1) // NUM_WORKERS  # Ceil division
    qids_chunks = [qids[i:i + chunk_size] for i in range(0, len(qids), chunk_size)]

    # Prepare arguments for each process
    process_args = [(qids, DOCUMENT_DATASET_OLD_INDEX_PATH) for qids in qids_chunks]

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        executor.map(process_qid, process_args)

    end_time = time.time()
    print(f"Finished processing in {(end_time - start_time) / 60} minutes.")
