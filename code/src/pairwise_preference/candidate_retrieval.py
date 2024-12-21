from tqdm import tqdm

import gzip
import json
import os
import matplotlib.pyplot as plt
import pyterrier as pt
from chatnoir_pyterrier import ChatNoirRetrieve, Feature


# Load the configuration settings
pwd = os.path.dirname(os.path.abspath(__file__))


def load_config(filename=pwd + "/../config.json"):
    with open(filename, "r") as f:
        config = json.load(f)
    return config


# Get the configuration settings
config = load_config()

ALL_QRELS = config['ALL_QRELS']
PER_QUERY = config['PER_QUERY']

TYPE_OLD = config['TYPE_OLD']
TYPE_NEW = config['TYPE_NEW']  # retrieve documents or passages from new dataset

# Either retrrieve with local index or with ChatNoir API
CHATNOIR_RETRIEVAL = config['CHATNOIR_RETRIEVAL']
CHATNOIR_INDICES = config['CHATNOIR_INDICES']

DOCUMENT_DATASET_NEW_NAME = config['DOCUMENT_DATASET_NEW_NAME']
DOCUMENT_DATASET_OLD_NAME = config['DOCUMENT_DATASET_OLD_NAME']
DOCUMENT_DATASET_NEW_NAME_PYTERRIER = config['DOCUMENT_DATASET_NEW_NAME_PYTERRIER']
DOCUMENT_DATASET_OLD_NAME_PYTERRIER = config['DOCUMENT_DATASET_OLD_NAME_PYTERRIER']
DOCUMENT_DATASET_NEW_NAME_PYTHON_API = config['DOCUMENT_DATASET_NEW_NAME_PYTHON_API']
DOCUMENT_DATASET_OLD_NAME_PYTHON_API = config['DOCUMENT_DATASET_OLD_NAME_PYTHON_API']

NEW_PATH = os.path.join(config['DATA_PATH'], DOCUMENT_DATASET_NEW_NAME)
OLD_PATH = os.path.join(config['DATA_PATH'], DOCUMENT_DATASET_OLD_NAME)

if TYPE_NEW == 'document':
    DATASET_NEW_INDEX_PATH = os.path.join(NEW_PATH, config['DOCUMENT_DATASET_NEW_INDEX_PATH'])
if TYPE_NEW == 'passage':
    DATASET_NEW_INDEX_PATH = os.path.join(NEW_PATH, config['PASSAGE_DATASET_NEW_INDEX_PATH'])

PASSAGE_DATASET_OLD_INDEX_PATH = os.path.join(OLD_PATH, config['PASSAGE_DATASET_OLD_INDEX_PATH'])

PASSAGE_DATASET_OLD_PATH = os.path.join(OLD_PATH, config['PASSAGE_DATASET_OLD_PATH'])

if CHATNOIR_RETRIEVAL:
    CANDIDATE_PATH = os.path.join(NEW_PATH, config['CANDIDATE_CHATNOIR_PATH'])
else:
    CANDIDATES_PATH = os.path.join(NEW_PATH, config['CANDIDATES_LOCAL_PATH'])

RECALL_PRECISION_PATH = os.path.join(NEW_PATH, 'recall_precision.txt')
APPROACHES = config['CANDIDATE_APPROACHES']

PASSAGE_ID_SEPARATOR = config['PASSAGE_ID_SEPARATOR']
KEY_SEPARATOR = config['KEY_SEPARATOR']
METRICS = config['METRICS']

# Caches
passages_text_cache = {}
queries_relevant_passages = {}

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
    with gzip.open(PASSAGE_DATASET_OLD_PATH, 'rt', encoding='UTF-8') as file:
        for line in file:
            line = json.loads(line)
            if line['docno'] not in known_passagenos:
                known_passagenos.add(line['docno'])
                yield {'docno': line['docno'], 'text': line['text']}


##########
# ORACLE #
##########


# All already judged documents (those that are in the qrels)
# INFO: only possible for old dataset
# Iterating over passage scores because already chunked in passages
def oracle_retrieval():
    qid_docnos = {}

    dataset = pt.get_dataset(DOCUMENT_DATASET_OLD_NAME_PYTERRIER)
    qrels = dataset.get_qrels(variant='relevance')
    for index, row in tqdm(qrels.iterrows(), desc='Caching qrels', unit='qrel'):
        if row['qid'] not in qid_docnos:
            qid_docnos[row['qid']] = []
        qid_docnos[row['qid']] += [row['docno']]

    return qid_docnos


######################
# APPROACH 1 - NAIVE #
######################


# HELPER for APPROACH 1
# Document yield function for indexing without duplicates
def yield_docs(dataset):
    known_docnos = set()
    for i in dataset.irds_ref().docs_iter():
        if i.doc_id not in known_docnos:
            known_docnos.add(i.doc_id)
            yield {'docno': i.doc_id, 'text': i.default_text()}


# For each query, retrieve top 2000 documents with bm25
# Cache of reults to reduce computation time in APPROACH 3
qid_docnos_naive_retrieval = {}


def naive_retrieval():
    dataset = pt.get_dataset(DOCUMENT_DATASET_NEW_NAME_PYTERRIER)

    # Retrieve top 2000 documents for each query
    if CHATNOIR_RETRIEVAL:
        chatnoir = ChatNoirRetrieve(index=CHATNOIR_INDICES, num_results=2000, retrieval_system="bm25")
    else:
        # Index datasetl
        if not os.path.exists(DATASET_NEW_INDEX_PATH):
            indexer = pt.IterDictIndexer(DATASET_NEW_INDEX_PATH)
            index_ref = indexer.index(yield_docs(dataset),
                                      meta={'docno': 50, 'text': 20000})
        else:
            index_ref = pt.IndexRef.of(DATASET_NEW_INDEX_PATH + '/data.properties')

        dataset_index = pt.IndexFactory.of(index_ref)

        bm25 = pt.terrier.Retriever(dataset_index, wmodel='BM25', num_results=2000)

    for query in tqdm(dataset.irds_ref().queries_iter(),
                      desc='Retrieving naive top documents',
                      unit='query'):
        qid = query.query_id
        query_text = query.default_text()
        if CHATNOIR_RETRIEVAL:
            query_results = chatnoir.search(query_text).loc[:, ['qid', 'docno']].head(2000)
        else:
            query_results = bm25.search(pt_tokenize(query_text), ).loc[:, ['qid', 'docno']].head(2000)

        qid_docnos_naive_retrieval[qid] = query_results['docno'].tolist()

    return qid_docnos_naive_retrieval


##################################
# APPROACH 2 - NEAREST NEIGHBOUR #
##################################


# HELPER for APPROACH 2: get for all queries the best passages in dictionary
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


# HELPER for APPROACH 2
# Get for each query all relevant passages in dictionary format qid: [passageno]
docno_passagenos = {}
with gzip.open(PASSAGE_DATASET_OLD_PATH, 'rt', encoding='UTF-8') as file:
    for line in tqdm(file, desc='Caching passages', unit='passage'):
        line = json.loads(line)
        docno, passageno = line['docno'].split(PASSAGE_ID_SEPARATOR)
        if docno not in docno_passagenos:
            docno_passagenos[docno] = []
        docno_passagenos[docno] += [line['docno']]

dataset = pt.get_dataset(DOCUMENT_DATASET_OLD_NAME_PYTERRIER)
qrels = dataset.get_qrels(variant='relevance')
for index, row in tqdm(qrels.iterrows(), desc='Caching qrels', unit='qrel'):
    if row['label'] > 0:
        if row['qid'] not in queries_relevant_passages:
            queries_relevant_passages[row['qid']] = []

        passagenos = docno_passagenos[row['docno']]
        queries_relevant_passages[row['qid']] += passagenos


# HELPER for APPROACH 2: get all passages text in dictionary format docno: [{passageno: text}]
with gzip.open(PASSAGE_DATASET_OLD_PATH, 'rt', encoding='UTF-8') as file:
    for line in file:
        line = json.loads(line)
        passages_text_cache[line['docno']] = line['text']


# For top 20 most relevant passages for each query, retrieve top 10 documents with bm25
# Cache of reults to reduce computation time in APPROACH 3
qid_docnos_nearest_neighbor_retrieval = {}


def nearest_neighbor_retrieval():
    dataset = pt.get_dataset(DOCUMENT_DATASET_NEW_NAME_PYTERRIER)

    # Retrieve for each relevant passage for its corresponding qid the top 20 docnos
    if CHATNOIR_RETRIEVAL:
        chatnoir = ChatNoirRetrieve(index=CHATNOIR_INDICES, num_results=20, retrieval_system="bm25")
    else:
        # Index dataset
        if not os.path.exists(DATASET_NEW_INDEX_PATH):
            indexer = pt.IterDictIndexer(DATASET_NEW_INDEX_PATH)
            index_ref = indexer.index(yield_docs(dataset),
                                      meta={'docno': 50, 'text': 20000})
        else:
            index_ref = pt.IndexRef.of(DATASET_NEW_INDEX_PATH + '/data.properties')

        dataset_index = pt.IndexFactory.of(index_ref)

        bm25 = pt.terrier.Retriever(dataset_index, wmodel='BM25', num_results=20)

    for query in tqdm(dataset.irds_ref().queries_iter(),
                      desc='Retrieving nearest neighbor top documents',
                      unit='query'):
        qid = query.query_id

        rel_doc_ids = queries_relevant_passages[qid]

        for rel_doc_id in rel_doc_ids:
            if CHATNOIR_RETRIEVAL:
                query_results = chatnoir.search(passages_text_cache[rel_doc_id]).loc[:, ['qid', 'docno']].head(20)
            else:
                query_results = bm25.search(pt_tokenize(passages_text_cache[rel_doc_id]), ).loc[:, [
                    'qid', 'docno']].head(20)
            if qid not in qid_docnos_nearest_neighbor_retrieval:
                qid_docnos_nearest_neighbor_retrieval[qid] = []
            qid_docnos_nearest_neighbor_retrieval[qid] += query_results['docno'].tolist()

        # remove duplicates
        qid_docnos_nearest_neighbor_retrieval[qid] = list(set(qid_docnos_nearest_neighbor_retrieval[qid]))

    return qid_docnos_nearest_neighbor_retrieval


#######################
# APPROACH 3 -  UNION #
#######################


# For each query, retrieve top 2000 documents with bm25 +
# For top 20 most relevant passages for each query, retrieve top 10 documents with bm25
qid_docnos_union_retrieval = {}


def union_retrieval():

    # Check if the caches are empty
    if qid_docnos_naive_retrieval == {}:
        naive_retrieval()
    if qid_docnos_nearest_neighbor_retrieval == {}:
        nearest_neighbor_retrieval()

    # Combine the caches
    for key, value in qid_docnos_naive_retrieval.items():
        qid_docnos_union_retrieval[key] = value

    for key, value in qid_docnos_nearest_neighbor_retrieval.items():
        if key in qid_docnos_union_retrieval:
            qid_docnos_union_retrieval[key] = list(set(qid_docnos_union_retrieval[key] + value))
        else:
            qid_docnos_union_retrieval[key] = value

    return qid_docnos_union_retrieval


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
# Only for old dataset
def compute_recall_precision(qid_docnos_cache, filename=None):

    # 1. Number of relevant documents per query
    num_all_relevant_documents_per_query = {}
    num_retrieved_documents_per_query = {}
    num_retrieved_relevant_documents_per_query = {}

    dataset = pt.get_dataset(DOCUMENT_DATASET_OLD_NAME_PYTERRIER)
    qrels = dataset.get_qrels(variant='relevance')
    for index, row in qrels.iterrows():
        if row['label'] > 0:
            if row['qid'] not in num_all_relevant_documents_per_query:
                num_all_relevant_documents_per_query[row['qid']] = 0
            num_all_relevant_documents_per_query[row['qid']] += 1

    for qid, docnos in qid_docnos_cache.items():
        num_retrieved_documents_per_query[qid] = len(docnos)
        num_retrieved_relevant_documents_per_query[qid] = 0

        for docno in docnos:
            if docno in qrels[(qrels['qid'] == qid) & (qrels['label'] > 0)]['docno'].values:
                num_retrieved_relevant_documents_per_query[qid] += 1

    # 2. Compute recall and precision for each query
    recalls = {qid: num_retrieved_relevant_documents_per_query[qid] / num_all_relevant_documents_per_query[qid]
               for qid in num_all_relevant_documents_per_query.keys()}
    precisions = {qid: num_retrieved_relevant_documents_per_query[qid] / num_retrieved_documents_per_query[qid]
                  for qid in num_all_relevant_documents_per_query.keys()}

    # 3. Plot the precision and recall for each query and save the plot as a PDF
    if filename:
        plot_precision_recall(recalls, precisions, filename=filename)

    # 4. Compute average recall and precision
    recall = sum(recalls.values()) / len(recalls)
    precision = sum(precisions.values()) / len(precisions)

    return recall, precision


def write_candidates(candidates_file, candidates, recall, precision):
    with gzip.open(candidates_file, 'wt', encoding='UTF-8') as file:
        for qid, docnos in candidates.items():
            file.write(json.dumps({
                "qid": qid,
                "docnos": docnos
            }) + '\n')

    with open(RECALL_PRECISION_PATH, 'a') as recall_precision_file:
        recall_precision_file.write(json.dumps({
            "approach_name": candidates_file.split('/')[-1].split('.')[0],
            "recall": recall,
            "precision": precision
        }) + '\n')


if __name__ == '__main__':
    # Reset recall and precision file
    with open(RECALL_PRECISION_PATH, 'w') as recall_precision_file:
        recall_precision_file.write('')

    # Recall and Precision for each approach
    docnos_naive = naive_retrieval()
    docnos_nearest_neighbor = nearest_neighbor_retrieval()
    docnos_union = union_retrieval()

    # # Print the number of documents (docnos) for each approach
    print(f'Naive: {sum([len(docnos) for docnos in docnos_naive.values()])} documents (docnos)')
    print(f'Nearest Neighbor: {sum([len(docnos) for docnos in docnos_nearest_neighbor.values()])} documents (docnos)')
    print(f'Union: {sum([len(docnos) for docnos in docnos_union.values()])} documents (docnos)')

    recall_naive, precision_naive = compute_recall_precision(docnos_naive, filename='recall_precision_naive.pdf')
    recall_nearest_neighbor, precision_nearest_neighbor = compute_recall_precision(
        docnos_nearest_neighbor, filename='recall_precision_nearest_neighbor.pdf')
    recall_union, precision_union = compute_recall_precision(docnos_union, filename='recall_precision_union.pdf')

    # print(f'Oracle: Recall={recall_oracle}, Precision={precision_oracle}')
    print(f'Naive: Recall={recall_naive}, Precision={precision_naive}')
    print(f'Nearest Neighbor: Recall={recall_nearest_neighbor}, Precision={precision_nearest_neighbor}')
    print(f'Union: Recall={recall_union}, Precision={precision_union}')

    # Write results to file
    naive_file_name = os.path.join(CANDIDATES_PATH, 'naive.jsonl.gz')
    nearest_neighbor_file_name = os.path.join(CANDIDATE_PATH, 'nearest_neighbor.jsonl.gz')
    union_file_name = os.path.join(CANDIDATE_PATH, 'union.jsonl.gz')

    write_candidates(naive_file_name, docnos_naive,
                     recall_naive, precision_naive)

    write_candidates(nearest_neighbor_file_name, docnos_nearest_neighbor,
                     recall_nearest_neighbor, precision_nearest_neighbor)

    write_candidates(union_file_name, docnos_union,
                     recall_union, precision_union)

