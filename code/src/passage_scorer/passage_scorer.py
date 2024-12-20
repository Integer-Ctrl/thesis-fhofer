import json
import os
import pyterrier as pt
from tqdm import tqdm
import gzip
from trectools import TrecQrel, TrecRun, TrecEval
import sys
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

# Script should only compute passage scores for one qid at a time
if len(sys.argv) < 2:
    print("Please provide a QID as an argument.")
    sys.exit(1)

JOB_ID = int(sys.argv[1])
QID = None

# Read qrels and cache relevant qrels
dataset = pt.get_dataset(DOCUMENT_DATASET_OLD_NAME_PYTERRIER)
qrels = dataset.get_qrels(variant='relevance')
qrels_cache = {}
for index, row in tqdm(qrels.iterrows(), desc='Caching qrels', unit='qrel'):
    if row['qid'] not in qrels_cache:
        qrels_cache[row['qid']] = qrels.loc[
            (qrels['qid'] == row['qid'])
        ].rename(columns={'qid': 'query', 'docno': 'docid', 'label': 'rel'})  # Rename columns
        qrels_cache[row['qid']]['query'] = 0  # Dummy value to enable merge of run and qrels (TrecEval)

keys = list(qrels_cache.keys())
if JOB_ID > len(keys):
    print("Job ID is out of range. Nothing to do.")
    sys.exit(0)
QID = keys[JOB_ID - 1]

# Initialize PyTerrier and Tokenizer
if not pt.java.started():
    pt.java.init()
tokeniser = pt.java.autoclass('org.terrier.indexing.tokenisation.Tokeniser').getTokeniser()


# Tokenize text
def pt_tokenize(text):
    return ' '.join(tokeniser.getTokens(text))


# Document yield function for indexing without duplicates
def yield_docs(dataset):
    known_docnos = set()
    for i in dataset.irds_ref().docs_iter():
        if i.doc_id not in known_docnos:
            known_docnos.add(i.doc_id)
            yield {'docno': i.doc_id, 'text': i.default_text()}


# Index dataset
if not os.path.exists(DOCUMENT_DATASET_OLD_INDEX_PATH):
    indexer = pt.IterDictIndexer(DOCUMENT_DATASET_OLD_INDEX_PATH)
    index_ref = indexer.index(yield_docs(dataset),
                              meta={'docno': 50, 'text': 20000})
else:
    index_ref = pt.IndexRef.of(DOCUMENT_DATASET_OLD_INDEX_PATH + '/data.properties')

dataset_index = pt.IndexFactory.of(index_ref)

# Read passages and cache them
passages_cache = {}
with gzip.open(PASSAGE_DATASET_OLD_PATH, 'rt', encoding='UTF-8') as file:
    for line in tqdm(file, desc='Caching passages', unit='passage'):
        line = json.loads(line)
        docno, passageno = line['docno'].split(PASSAGE_ID_SEPARATOR)
        if docno not in passages_cache:
            passages_cache[docno] = []
        passages_cache[docno] += [line]

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


# Write passage scores to file

def process_qid(qid):

    results = []
    relevant_results = []

    print(f"Processing QID {qid} in process with Job: {JOB_ID}")
    for docno in qrels_cache[qid]['docid']:
        for passage in passages_cache[docno]:
            # wod = without original document
            qrels_for_query = get_qrels_for_query(qid, include_original_document=True)
            qrels_for_query_wod = get_qrels_for_query(qid, include_original_document=False)

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

    return results, relevant_results


if __name__ == '__main__':
    start_time = time.time()

    relevant_path = os.path.join(PASSAGE_DATASET_OLD_SCORE_REL_PATH, f"qid_{QID}.jsonl.gz")
    all_path = os.path.join(PASSAGE_DATASET_OLD_SCORE_AQ_PATH, f"qid_{QID}.jsonl.gz")

    # Check if scores already exist
    if os.path.exists(relevant_path) and os.path.exists(all_path):
        print(f"Scores for QID {QID} already exist. Exiting.")
        sys.exit(0)

    results, relevant_results = process_qid(QID)

    end_time = time.time()
    print(f"Job {JOB_ID} finished processing QID {QID} in {(end_time - start_time) / 60} minutes.")

    with gzip.open(relevant_path, 'wt', encoding='UTF-8') as relevant_qrels_file:
        for scores in relevant_results:
            relevant_qrels_file.write(json.dumps(scores) + '\n')

    with gzip.open(all_path, 'wt', encoding='UTF-8') as all_qrels_file:
        for scores in results:
            all_qrels_file.write(json.dumps(scores) + '\n')
