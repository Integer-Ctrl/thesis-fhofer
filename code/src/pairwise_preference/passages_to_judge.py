from tqdm import tqdm

import gzip
import json
import ir_datasets
import os

import pyterrier as pt


# Load the configuration settings
# def load_config(filename="../config.json"): does not work with debug
def load_config(filename="/mnt/ceph/storage/data-tmp/current/ho62zoq/thesis-fhofer/code/src/config.json"):
    with open(filename, "r") as f:
        config = json.load(f)
    return config


# Get the configuration settings
config = load_config()

ALL_QRELS = config['ALL_QRELS']
PER_QUERY = config['PER_QUERY']

DOCUMENT_DATASET_NAME = config['DOCUMENT_DATASET_NAME']
DOCUMENT_DATASET_NAME_PYTERRIER = config['DOCUMENT_DATASET_NAME_PYTERRIER']
DOCUMENT_DATASET_NAME_PYTHON_API = config['DOCUMENT_DATASET_NAME_PYTHON_API']

DATA_PATH = os.path.join(config['DATA_PATH'], DOCUMENT_DATASET_NAME)
PASSAGE_DATASET_INDEX_PATH = os.path.join(DATA_PATH, config['PASSAGE_DATASET_INDEX_PATH'])

PASSAGE_DATASET_PATH = os.path.join(DATA_PATH, config['PASSAGE_DATASET_PATH'])

if ALL_QRELS:
    PASSAGE_DATASET_SCORE_PATH = os.path.join(DATA_PATH, config['PASSAGE_DATASET_SCORE_AQ_PATH'])
    if PER_QUERY:
        PASSAGES_TO_DOCUMENT_CORRELATION_SCORE_PATH = os.path.join(
            DATA_PATH, config['PASSAGES_TO_DOCUMENT_CORRELATION_SCORE_PQ_AQ_PATH'])
    else:
        PASSAGES_TO_DOCUMENT_CORRELATION_SCORE_PATH = os.path.join(
            DATA_PATH, config['PASSAGES_TO_DOCUMENT_CORRELATION_SCORE_AQ_PATH'])
else:
    PASSAGE_DATASET_SCORE_PATH = os.path.join(DATA_PATH, config['PASSAGE_DATASET_SCORE_PATH'])
    if PER_QUERY:
        PASSAGES_TO_DOCUMENT_CORRELATION_SCORE_PATH = os.path.join(
            DATA_PATH, config['PASSAGES_TO_DOCUMENT_CORRELATION_SCORE_PQ_PATH'])
    else:
        PASSAGES_TO_DOCUMENT_CORRELATION_SCORE_PATH = os.path.join(
            DATA_PATH, config['PASSAGES_TO_DOCUMENT_CORRELATION_SCORE_PATH'])

PASSAGES_TO_JUDGE_APPROACH1_PATH = os.path.join(DATA_PATH, config['PASSAGES_TO_JUDGE_APPROACH1_PATH'])
PASSAGES_TO_JUDGE_APPROACH2_PATH = os.path.join(DATA_PATH, config['PASSAGES_TO_JUDGE_APPROACH2_PATH'])
PASSAGES_TO_JUDGE_APPROACH3_PATH = os.path.join(DATA_PATH, config['PASSAGES_TO_JUDGE_APPROACH3_PATH'])

PASSAGE_ID_SEPARATOR = config['PASSAGE_ID_SEPARATOR']
KEY_SEPARATOR = config['KEY_SEPARATOR']
METRICS = config['METRICS']

# Initialize PyTerrier and Tokenizer
if not pt.java.started():
    pt.java.init()
tokeniser = pt.java.autoclass('org.terrier.indexing.tokenisation.Tokeniser').getTokeniser()


# Tokenize text
def pt_tokenize(text):
    return ' '.join(tokeniser.getTokens(text))


# Passage yield function for indexing
def yield_passages():
    known_passagenos = set()
    with gzip.open(PASSAGE_DATASET_PATH, 'rt', encoding='UTF-8') as file:
        for line in file:
            line = json.loads(line)
            if line['docno'] not in known_passagenos:
                known_passagenos.add(line['docno'])
                yield {'docno': line['docno'], 'text': line['text']}


# HELPER for APPROACH 3: get type of best scoring method in rank correlation
def get_best_scoring_methods():
    with gzip.open(PASSAGES_TO_DOCUMENT_CORRELATION_SCORE_PATH, 'rt', encoding='UTF-8') as file:
        for line in file:  # already decending sorted
            return json.loads(line)  # return only best scoring method


# HELPER for APPROACH 3: get all passage scores in dictionary format qid: {docno: score}
# just score of the best scoring method
def get_passages_scores(cache, metric):
    with gzip.open(PASSAGE_DATASET_SCORE_PATH, 'rt', encoding='UTF-8') as file:
        for line in tqdm(file, desc='Caching passage scores', unit='passage'):
            data = json.loads(line)
            qid = data['qid']        # Extract query ID
            docno = data['docno']    # Extract document number

            # Store the best score in the cache
            if qid not in cache:
                cache[qid] = {}
            cache[qid][docno] = data[metric]


# HELPER for APPROACH 3: get for all queries the best passages in dictionary
# format query_id: [docno] without duplicates docno
def get_queries_best_passages_one_per_document(cache, scores):
    for qid, passageno_scores in scores.items():
        # Step 1: Parse docnos and sort by score
        docnos_best_passagenos = {}
        for passageno, score in passageno_scores.items():
            # Extract docno by removing the suffix ___x
            docno, _ = passageno.split(PASSAGE_ID_SEPARATOR)
            # Keep the highest-scoring passageno for each docno
            if docno not in docnos_best_passagenos or score > docnos_best_passagenos[docno][1]:
                docnos_best_passagenos[docno] = (passageno, score)

        # Step 2: Extract highest-scored passagenos and sort them in descending order
        best_passagenos = [item[0]
                           for item in sorted(docnos_best_passagenos.values(), key=lambda x: x[1], reverse=True)]

        # Add to result
        cache[qid] = best_passagenos


# HELPER for APPROACH 3: get all passages text in dictionary format docno: [{passageno: text}]
def get_passages_text(cache):
    with gzip.open(PASSAGE_DATASET_PATH, 'rt', encoding='UTF-8') as file:
        for line in file:
            line = json.loads(line)
            cache[line['docno']] = line['text']


# APPROACH 1: All already scored passages (those that are in the qrels)
# INFO: only possible for old dataset
# Iterating over passage scores because already chunked in passages
def get_qrels_passages():
    qid_docnos = {}

    with gzip.open(PASSAGE_DATASET_SCORE_PATH, 'rt', encoding='UTF-8') as file:
        for line in tqdm(file, desc='Caching passage scores', unit='passage'):
            data = json.loads(line)
            qid = data['qid']        # Extract query ID
            docno = data['docno']    # Extract document number

            # Store the best score in the cache
            if qid not in qid_docnos:
                qid_docnos[qid] = []
            qid_docnos[qid] += [docno]

    return qid_docnos


# APPROACH 2: For each query, retrieve top 2000 passages with bm25
def get_top_passages_for_queries():
    # Index passage dataset
    if not os.path.exists(PASSAGE_DATASET_INDEX_PATH):
        indexer = pt.IterDictIndexer(PASSAGE_DATASET_INDEX_PATH)
        index_ref = indexer.index(yield_passages(),
                                  meta={'docno': 50, 'text': 20000})
    else:
        index_ref = pt.IndexRef.of(PASSAGE_DATASET_INDEX_PATH + '/data.properties')

    passage_index = pt.IndexFactory.of(index_ref)

    # Retrieve top 2000 passages for each query
    # TODO: refactor retrieval with ChatNoir API
    bm25 = pt.terrier.Retriever(passage_index, wmodel='BM25')
    queries_top_passages = {}
    for query in tqdm(ir_datasets.load(DOCUMENT_DATASET_NAME_PYTHON_API).queries_iter(),
                      desc='Retrieving top passages',
                      unit='query'):
        qid = query.query_id
        query_text = query.default_text()
        # BUG: only 1000 passages can be retreived
        query_results = bm25.search(pt_tokenize(query_text), ).loc[:, ['qid', 'docno']].head(2000)
        queries_top_passages[qid] = query_results['docno'].tolist()

    return queries_top_passages


# APPROACH 3: For each query, retrieve top 2000 passages with bm25 +
#             for top 20 most relevant passages for each query, retrieve top 10 passages with bm25
def get_top_passages_for_queries_advanced():
    # Index passage dataset
    if not os.path.exists(PASSAGE_DATASET_INDEX_PATH):
        indexer = pt.IterDictIndexer(PASSAGE_DATASET_INDEX_PATH)
        index_ref = indexer.index(yield_passages(),
                                  meta={'docno': 50, 'text': 20000})
    else:
        index_ref = pt.IndexRef.of(PASSAGE_DATASET_INDEX_PATH + '/data.properties')

    passage_index = pt.IndexFactory.of(index_ref)

    # Retrieve top 2000 passages for each query
    # TODO: refactor retrieval with ChatNoir API
    bm25 = pt.terrier.Retriever(passage_index, wmodel='BM25')
    queries_top_passages = {}
    for query in tqdm(ir_datasets.load(DOCUMENT_DATASET_NAME_PYTHON_API).queries_iter(),
                      desc='Retrieving top passages',
                      unit='query'):
        qid = query.query_id
        query_text = query.default_text()
        # BUG: only 1000 passages can be retreived
        query_results = bm25.search(pt_tokenize(query_text), ).loc[:, ['qid', 'docno']].head(2000)
        queries_top_passages[qid] = query_results['docno'].tolist()

    # Retrieve top 10 passages for top 20 most relevant passages for each query
    passages_score_cache = {}
    queries_best_passages_cache = {}
    passages_text_cache = {}

    get_passages_text(passages_text_cache)
    best_scoring_method = get_best_scoring_methods()
    get_passages_scores(passages_score_cache, best_scoring_method['metric'])
    get_queries_best_passages_one_per_document(queries_best_passages_cache, passages_score_cache)

    rel_doc_ids = queries_best_passages_cache[qid]
    rel_doc_ids = rel_doc_ids[:20]

    for rel_doc_id in rel_doc_ids:
        query_results = bm25.search(pt_tokenize(passages_text_cache[rel_doc_id]), ).loc[:, ['qid', 'docno']].head(10)
        queries_top_passages[qid] += query_results['docno'].tolist()

    return queries_top_passages


# TODO Recall and Precision for each approach
passages_approach1 = get_qrels_passages()
passages_approach2 = get_top_passages_for_queries()
passages_approach3 = get_top_passages_for_queries_advanced()

# TODO Save results
