from tqdm import tqdm

import gzip
import json
import ir_datasets
import os
import matplotlib.pyplot as plt

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
PASSAGE_DATASET_RELEVANT_SCORE_PATH = os.path.join(DATA_PATH, config['PASSAGE_DATASET_SCORE_PATH'])

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

PASSAGES_TO_JUDGE_PATH = os.path.join(DATA_PATH, config['PASSAGES_TO_JUDGE_PATH'])
APPROACHES = config['PASSAGES_TO_JUDGE_APPROACHES']

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

    passages_score_cache = {}
    queries_best_passages_cache = {}
    passages_text_cache = {}

    get_passages_text(passages_text_cache)
    best_scoring_method = get_best_scoring_methods()
    get_passages_scores(passages_score_cache, best_scoring_method['metric'])
    get_queries_best_passages_one_per_document(queries_best_passages_cache, passages_score_cache)

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
        rel_doc_ids = queries_best_passages_cache[qid]
        rel_doc_ids = rel_doc_ids[:20]

        for rel_doc_id in rel_doc_ids:
            query_results = bm25.search(pt_tokenize(passages_text_cache[rel_doc_id]), ).loc[:, [
                'qid', 'docno']].head(10)
            queries_top_passages[qid] += query_results['docno'].tolist()
            # Remove duplicates
            queries_top_passages[qid] = list(set(queries_top_passages[qid]))

    return queries_top_passages


# Compute Recall and Precision for each approach
def compute_recall_precision_old(passages):
    # 1. get number of relevant passages in dataset
    #    passage is relevant if its docno is in the qrels
    num_all_relevant_passages = 0
    num_retrieved_passages = 0
    num_retrieved_relevant_passages = 0
    # 2. get number of relevant passages in retrieved passages
    with gzip.open(PASSAGE_DATASET_RELEVANT_SCORE_PATH, 'rt', encoding='UTF-8') as file:
        lines = file.readlines()
        num_all_relevant_passages = len(lines)

        relevant_docnos = []
        for line in lines:
            data = json.loads(line)
            relevant_docnos.append(data['docno'])

        for qid, docnos in passages.items():
            num_retrieved_passages += len(docnos)
            for docno in docnos:
                if docno in relevant_docnos:
                    num_retrieved_relevant_passages += 1

    # 3. compute recall and precision
    recall = num_retrieved_relevant_passages / num_all_relevant_passages
    precision = num_retrieved_relevant_passages / num_retrieved_passages

    return recall, precision


# Function to plot Precision and Recall for each query and optionally save to PDF
def plot_precision_recall(recalls, precisions, filename=None):
    # Check if a filename is provided
    if filename:
        # Prepare data
        queries = list(recalls.keys())
        recall_values = list(recalls.values())
        precision_values = list(precisions.values())

        # Create the plot
        plt.figure(figsize=(10, 8))
        plt.scatter(recall_values, precision_values, color='blue', s=100, alpha=0.7)

        # Annotate each point with its query ID
        for i, query in enumerate(queries):
            plt.text(recall_values[i], precision_values[i], query, fontsize=10, ha='right', va='bottom')

        # Set axis limits
        plt.xlim(0, 1)
        plt.ylim(0, 1)

        # Add labels and grid
        plt.title('Precision vs. Recall per Query', fontsize=16)
        plt.xlabel('Recall', fontsize=14)
        plt.ylabel('Precision', fontsize=14)
        plt.grid(True)

        # Tight layout for better spacing
        plt.tight_layout()

    # Save plot
        plt.savefig(filename, format='pdf')
        print(f"Plot saved to {filename}")


# Compute Recall and Precision for each approach
def compute_recall_precision(passages, filename=None):
    # Load relevant document IDs from the dataset
    relevant_docnos_per_query = {}
    with gzip.open(PASSAGE_DATASET_RELEVANT_SCORE_PATH, 'rt', encoding='UTF-8') as file:
        for line in file:
            data = json.loads(line)
            if data['qid'] not in relevant_docnos_per_query:
                relevant_docnos_per_query[data['qid']] = []
            relevant_docnos_per_query[data['qid']] += [data['docno']]

    num_all_relevant_passages_per_query = {qid: len(docnos) for qid, docnos in relevant_docnos_per_query.items()}
    num_retrieved_passages_per_query = {qid: len(docnos) for qid, docnos in passages.items()}
    num_retrieved_relevant_passages_per_query = {qid: 0 for qid in relevant_docnos_per_query.keys()}

    for qid, docnos in passages.items():
        for docno in docnos:
            if docno in relevant_docnos_per_query[qid]:
                num_retrieved_relevant_passages_per_query[qid] += 1

    # Compute recall and precision for each query
    recalls = {qid: num_retrieved_relevant_passages_per_query[qid] / num_all_relevant_passages_per_query[qid]
               for qid in relevant_docnos_per_query.keys()}
    precisions = {qid: num_retrieved_relevant_passages_per_query[qid] / num_retrieved_passages_per_query[qid]
                  for qid in relevant_docnos_per_query.keys()}

    # Plot the precision and recall for each query and save the plot as a PDF
    if filename:
        plot_precision_recall(recalls, precisions, filename=filename)

    # Compute average recall and precision
    recall = sum(recalls.values()) / len(recalls)
    precision = sum(precisions.values()) / len(precisions)

    return recall, precision


# TODO Recall and Precision for each approach
passages_approach1 = get_qrels_passages()
passages_approach2 = get_top_passages_for_queries()
passages_approach3 = get_top_passages_for_queries_advanced()

recall_approach1, precision_approach_1 = compute_recall_precision(passages_approach1, 'recall_precision_approach1.pdf')
recall_approach2, precision_approach_2 = compute_recall_precision(passages_approach2, 'recall_precision_approach2.pdf')
recall_approach3, precision_approach_3 = compute_recall_precision(passages_approach3, 'recall_precision_approach3.pdf')

print('Approach 1:')
print(f'Recall: {recall_approach1}, Precision: {precision_approach_1}')
print('Approach 2:')
print(f'Recall: {recall_approach2}, Precision: {precision_approach_2}')
print('Approach 3:')
print(f'Recall: {recall_approach3}, Precision: {precision_approach_3}')


# Write results to file
with gzip.open(PASSAGES_TO_JUDGE_PATH, 'wt', encoding='UTF-8') as file:
    for approach in APPROACHES:
        if approach == 'approach1':
            file.write(json.dumps({
                "approach_name": approach,
                "recall": recall_approach1,
                "precision": precision_approach_1,
                "judge": passages_approach1
            }) + '\n')
        elif approach == 2:
            file.write(json.dumps({
                "approach_name": approach,
                "recall": recall_approach2,
                "precision": precision_approach_2,
                "judge": passages_approach2
            }) + '\n')
        elif approach == 3:
            file.write(json.dumps({
                "approach_name": approach,
                "recall": recall_approach3,
                "precision": precision_approach_3,
                "judge": passages_approach3
            }) + '\n')
