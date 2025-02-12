from tqdm import tqdm
import gzip
import json
import os
import matplotlib.pyplot as plt
import ir_datasets
import pyterrier as pt
from chatnoir_pyterrier import ChatNoirRetrieve
import re
from glob import glob
import pandas as pd
from spacy_passage_chunker import SpacyPassageChunker
import sys
from ir_datasets_clueweb22 import register
import time

register()


# Load the configuration settings
pwd = os.path.dirname(os.path.abspath(__file__))


def load_config(filename=pwd + "/../config.json"):
    with open(filename, "r") as f:
        config = json.load(f)
    return config


# Get the configuration settings
config = load_config()

# Either retrrieve with local index or with ChatNoir API
CHATNOIR_RETRIEVAL = config['CHATNOIR_RETRIEVAL']
CHATNOIR_TARGET_INDICES = config['CHATNOIR_TARGET_INDICES']
CHATNOIR_API_KEY = config['CHATNOIR_API_KEY']

DOCUMENT_DATASET_TARGET_NAME = config['DOCUMENT_DATASET_TARGET_NAME']
DOCUMENT_DATASET_SOURCE_NAME = config['DOCUMENT_DATASET_SOURCE_NAME']
DOCUMENT_DATASET_SOURCE_NAME_PYTERRIER = config['DOCUMENT_DATASET_SOURCE_NAME_PYTERRIER']
DOCUMENT_DATASET_TARGET_NAME_PYTHON_API = config['DOCUMENT_DATASET_TARGET_NAME_PYTHON_API']

SOURCE_PATH = os.path.join(config['DATA_PATH'], DOCUMENT_DATASET_SOURCE_NAME)
TARGET_PATH = os.path.join(SOURCE_PATH, config["DOCUMENT_DATASET_TARGET_NAME"])

# For how many queries the candidates should be created - None for all queries
QUERIES = config['QUERIES']  # retrieve for following queries only

# Securety check
if DOCUMENT_DATASET_TARGET_NAME == 'clueweb22/b':
    if CHATNOIR_RETRIEVAL is False:
        print("Target dataset is clueWeb22/b. Please use ChatNoir for retrieval.")
        exit()
    if QUERIES is None:
        print("Target dataset is clueWeb22/b. Please specify the number of queries to process.")
        exit()
    print(f"Running candidate retrieval for {DOCUMENT_DATASET_SOURCE_NAME} on target clueWeb22/b")

DOCUMENT_DATASET_SOURCE_INDEX_PATH = os.path.join(SOURCE_PATH, config['DOCUMENT_DATASET_SOURCE_INDEX_PATH'])

PASSAGE_DATASET_SOURCE_PATH = os.path.join(SOURCE_PATH, config['PASSAGE_DATASET_SOURCE_PATH'])
# Pattern to match the files
PASSAGE_DATASET_SOURCE_SCORE_REL_PATH = os.path.join(SOURCE_PATH, config['PASSAGE_DATASET_SOURCE_SCORE_REL_PATH'])
FILE_PATTERN = os.path.join(PASSAGE_DATASET_SOURCE_SCORE_REL_PATH, "qid_*.jsonl.gz")
# Regular expression to extract the number
NUMBER_PATTERN = re.compile(r"qid_(\d+)\.jsonl\.gz")

if CHATNOIR_RETRIEVAL:
    CANDIDATES_PATH = os.path.join(TARGET_PATH, config['CANDIDATE_CHATNOIR_PATH'])
else:
    CANDIDATES_PATH = os.path.join(TARGET_PATH, config['CANDIDATES_LOCAL_PATH'])
print(f"Save location: {CANDIDATES_PATH}")

RECALL_PRECISION_PATH = os.path.join(CANDIDATES_PATH, 'recall_precision.txt')

PASSAGE_SCORES_CROSS_VALIDATION_SCORES_PATH = os.path.join(SOURCE_PATH, config['CROSS_VALIDATION_SCORES_PATH'])

PASSAGE_ID_SEPARATOR = config['PASSAGE_ID_SEPARATOR']
KEY_SEPARATOR = config['KEY_SEPARATOR']

# Initialize PyTerrier and Tokenizer
if not pt.java.started():
    pt.java.init()
tokeniser = pt.java.autoclass('org.terrier.indexing.tokenisation.Tokeniser').getTokeniser()


# Tokenize text
def pt_tokenize(text):
    return ' '.join(tokeniser.getTokens(text))


# Access the target dataset for candidate creation
target_docno_passagenos = {}
target_passages_text_cache = {}


###################
# PASSAGE CHUNKER #
###################

class PassageChunker:

    def __init__(self):
        self.dataset = ir_datasets.load(DOCUMENT_DATASET_TARGET_NAME_PYTHON_API)
        self.docstore = self.dataset.docs_store()
        self.chunker = SpacyPassageChunker()

    def chunk_batch(self, batch):
        # Chunk the batch of documents
        chunked_batch = self.chunker.process_batch(batch)

        for chunked_doc in chunked_batch:
            # Add to dictionaries
            if chunked_doc['docno'] not in target_docno_passagenos:
                target_docno_passagenos[chunked_doc['docno']] = []
                target_passages_text_cache[chunked_doc['docno']] = {}

            for passage in chunked_doc['contents']:
                passage_id = chunked_doc['docno'] + PASSAGE_ID_SEPARATOR + str(passage['id'])
                target_docno_passagenos[chunked_doc['docno']] += [passage_id]
                target_passages_text_cache[chunked_doc['docno']][passage_id] = passage['body']

    def chunk_target_documents(self, docs_to_chunk, batch_size=1000):

        BATCH_SIZE = batch_size
        batch = []
        known_doc_ids = set()
        chunked_docs_count = 0

        docs_dict = self.docstore.get_many(docs_to_chunk)
        print(f"Loaded {len(docs_dict)} documents from {len(docs_to_chunk)} docs to chunk")

        print(f"Chunking documents: {chunked_docs_count}")
        for docid, doc in docs_dict.items():
            # Skip documents that should not be chunked
            if doc.doc_id not in docs_to_chunk:
                continue

            # Skip documents that have already been processed
            if doc.doc_id in known_doc_ids:
                continue
            known_doc_ids.add(doc.doc_id)

            # Format the document
            formatted_doc = {
                'docno': doc.doc_id,
                'contents': doc.default_text()
            }

            # Add the document to the current batch
            batch.append(formatted_doc)

            # If the batch reaches the specified batch size, process and save it
            if len(batch) >= BATCH_SIZE:
                chunked_docs_count += len(batch)
                print(f"Chunking documents: {chunked_docs_count}")
                self.chunk_batch(batch)
                # Reset the batch after saving
                batch = []

        # Process and save any remaining documents in the batch
        if batch:
            chunked_docs_count += len(batch)
            self.chunk_batch(batch)

        print(f"Chunked {chunked_docs_count} documents")


###################################################
# GET PASSAGES FOR PAIRWISE PREFERENCE CANDIDATES #
###################################################

# Get type of metric with highest rank correlation
best_scoring_metric_retriever = None
with gzip.open(PASSAGE_SCORES_CROSS_VALIDATION_SCORES_PATH, 'rt', encoding='UTF-8') as file:
    for line in file:  # already decending sorted
        data = json.loads(line)  # return only best scoring method
        best_scoring_metric = data['eval_method___retriever___metric'].split(KEY_SEPARATOR)[-1]
        best_scoring_retriever = data['eval_method___retriever___metric'].split(KEY_SEPARATOR)[1]
        best_scoring_metric_retriever = best_scoring_metric + '_' + best_scoring_retriever
        break

# Get all passage scores in dictionary format qid: {docno: score} # just score of the best scoring method
passages_score_cache = {}
for file_path in glob(FILE_PATTERN):
    with gzip.open(file_path, 'rt', encoding='UTF-8') as file:
        for line in file:
            data = json.loads(line)
            qid = data['qid']        # Extract query ID
            docno = data['docno']    # Extract document number

            # Store the best score in the passages_score_cache
            if qid not in passages_score_cache:
                passages_score_cache[qid] = {}
            passages_score_cache[qid][docno] = data[best_scoring_metric_retriever]

# Get for all queries the best and worst passages in dictionary format query_id: [docno] without duplicates docno
queries_best_passages_cache = {}  # multiple passages of one document possible
queries_worst_passages_cache = {}  # multiple passages of one document possible
queries_best_passages_opd_cache = {}  # opd = one per documnet, maximum of one passage per document
queries_worst_passages_opd_cache = {}  # opd = one per documnet, maximum of one passage per document

for qid, passageno_scores in passages_score_cache.items():
    # Parse docnos and sort by score
    docnos_best_passagenos_opd = {}
    docnos_worst_passagenos_opd = {}
    for passageno, score in passageno_scores.items():
        # Extract docno by removing the suffix ___x
        docno, _ = passageno.split(PASSAGE_ID_SEPARATOR)

        # Keep the highest-scoring passageno for each docno for opd approach
        if docno not in docnos_best_passagenos_opd or score > docnos_best_passagenos_opd[docno][1]:
            docnos_best_passagenos_opd[docno] = (passageno, score)

        # Keep the lowest-scoring passageno for each docno for opd approach
        if docno not in docnos_worst_passagenos_opd or score < docnos_worst_passagenos_opd[docno][1]:
            docnos_worst_passagenos_opd[docno] = (passageno, score)

    # Sort by score descending
    queries_best_passages_cache[qid] = [item[0]
                                        for item in sorted(passageno_scores.items(), key=lambda x: x[1], reverse=True)]

    # Sort by score ascending
    queries_worst_passages_cache[qid] = [item[0]
                                         for item in sorted(passageno_scores.items(), key=lambda x: x[1])]
    # opd: Extract highest-scored passagenos and sort them in descending order
    best_passagenos = [item[0]
                       for item in sorted(docnos_best_passagenos_opd.values(), key=lambda x: x[1], reverse=True)]
    queries_best_passages_opd_cache[qid] = best_passagenos

    # opd: Extract lowest-scored passagenos and sort them in ascending order
    worst_passagenos = [item[0]
                        for item in sorted(docnos_best_passagenos_opd.values(), key=lambda x: x[1])]
    queries_worst_passages_opd_cache[qid] = worst_passagenos


# Get for each query all relevant passages in dictionary format qid: [passageno]
source_docno_passagenos = {}
# Get all passages text in dictionary format docno: {passageno: text}
source_passages_text_cache = {}
# Get all relevant passages in dictionary format qid: [passageno]
queries_relevant_passagenos = {}

with gzip.open(PASSAGE_DATASET_SOURCE_PATH, 'rt', encoding='UTF-8') as file:
    for line in file:
        line = json.loads(line)
        docno, passageno = line['docno'].split(PASSAGE_ID_SEPARATOR)
        if docno not in source_docno_passagenos:
            source_docno_passagenos[docno] = []
            source_passages_text_cache[docno] = {}
        source_docno_passagenos[docno] += [line['docno']]
        source_passages_text_cache[docno][line['docno']] = line['text']

######################
# APPROACH 1 - NAIVE #
######################


# For each query, retrieve top 2000 documents with bm25
def naive_retrieval(num_retrieval_docs):
    qid_docnos_naive_retrieval = {}
    dataset = pt.get_dataset(DOCUMENT_DATASET_SOURCE_NAME_PYTERRIER)

    # Retrieve top 2000 documents for each query
    # 1000 documents via the query text and 1000 documents via the query description
    if CHATNOIR_RETRIEVAL:
        chatnoir = ChatNoirRetrieve(api_key=CHATNOIR_API_KEY,
                                    index=CHATNOIR_TARGET_INDICES,
                                    search_method="bm25",
                                    num_results=num_retrieval_docs)
    else:
        index_ref = pt.IndexRef.of(DOCUMENT_DATASET_SOURCE_INDEX_PATH + '/data.properties')
        dataset_index = pt.IndexFactory.of(index_ref)

        bm25 = pt.terrier.Retriever(dataset_index, wmodel='BM25', num_results=num_retrieval_docs)

    for query in tqdm(dataset.irds_ref().queries_iter(),
                      desc='Retrieving naive top documents',
                      unit='query'):

        qid = query.query_id
        # Only retrieve for the specified queries
        if QUERIES is not None and qid not in QUERIES:
            continue
        if DOCUMENT_DATASET_SOURCE_NAME == 'disks45/nocr/trec-robust-2004' and qid == '672':
            continue  # Skip query 672 as it has no relevant passages

        query_text = query.default_text()
        query_description = query.description if hasattr(query, 'description') else False

        if CHATNOIR_RETRIEVAL:
            query_results = chatnoir.search(query_text).loc[:, ['qid', 'docno']].head(num_retrieval_docs)
            # Sleep to prevent chatnoir internal server error
            time.sleep(5)
            if query_description:
                additional_results = chatnoir.search(query_description).loc[:, [
                    'qid', 'docno']].head(num_retrieval_docs)
                query_results = pd.concat([query_results, additional_results], ignore_index=True)
                # Sleep to prevent chatnoir internal server error
                time.sleep(5)
        else:
            query_results = bm25.search(pt_tokenize(query_text), ).loc[:, ['qid', 'docno']].head(num_retrieval_docs)
            if query_description:
                additional_results = bm25.search(pt_tokenize(query_description)).loc[:, [
                    'qid', 'docno']].head(num_retrieval_docs)
                query_results = pd.concat([query_results, additional_results], ignore_index=True)

        # Remove duplicates
        qid_docnos_naive_retrieval[qid] = list(set(query_results['docno'].tolist()))

    return qid_docnos_naive_retrieval


#################################
# APPROACH 2 - NEAREST NEIGHBOR #
#################################

# For each selected (passage chunker) relevant passages for each query, retrieve top 20 documents with bm25
def nearest_neighbor_retrieval(num_top_passages, num_retrieval_docs, one_per_document):
    qid_docnos_nearest_neighbor_retrieval = {}
    dataset = pt.get_dataset(DOCUMENT_DATASET_SOURCE_NAME_PYTERRIER)

    # Retrieve for each relevant passage for its corresponding qid the top 20 docnos
    if CHATNOIR_RETRIEVAL:  # Case if target is ClueWeb22/b
        chatnoir = ChatNoirRetrieve(api_key=CHATNOIR_API_KEY,
                                    index=CHATNOIR_TARGET_INDICES,
                                    search_method="bm25",
                                    num_results=num_retrieval_docs)
    else:  # Case if target is source dataset
        index_ref = pt.IndexRef.of(DOCUMENT_DATASET_SOURCE_INDEX_PATH + '/data.properties')
        dataset_index = pt.IndexFactory.of(index_ref)

        bm25 = pt.terrier.Retriever(dataset_index, wmodel='BM25', num_results=num_retrieval_docs)

    for query in tqdm(dataset.irds_ref().queries_iter(),
                      desc='Retrieving nearest neighbor top documents',
                      unit='query'):

        qid = query.query_id
        # Only retrieve for the specified queries
        if QUERIES is not None and qid not in QUERIES:
            continue
        if DOCUMENT_DATASET_SOURCE_NAME == 'disks45/nocr/trec-robust-2004' and qid == '672':
            continue  # Skip query 672 as it has no relevant passages

        if one_per_document:
            top_rel_doc_ids = queries_best_passages_opd_cache[qid][:num_top_passages]
        else:
            top_rel_doc_ids = queries_best_passages_cache[qid][:num_top_passages]

        for rel_doc_id in top_rel_doc_ids:
            docno, _ = rel_doc_id.split(PASSAGE_ID_SEPARATOR)
            if CHATNOIR_RETRIEVAL:
                query_results = chatnoir.search(
                    source_passages_text_cache[docno][rel_doc_id]).loc[:, ['qid', 'docno']].head(num_retrieval_docs)
                # Sleep to prevent chatnoir internal server error
                time.sleep(5)
            else:
                query_results = bm25.search(pt_tokenize(source_passages_text_cache[docno][rel_doc_id]), ).loc[:, [
                    'qid', 'docno']].head(num_retrieval_docs)
            if qid not in qid_docnos_nearest_neighbor_retrieval:
                qid_docnos_nearest_neighbor_retrieval[qid] = []
            qid_docnos_nearest_neighbor_retrieval[qid] += query_results['docno'].tolist()

        # remove duplicates
        qid_docnos_nearest_neighbor_retrieval[qid] = list(set(qid_docnos_nearest_neighbor_retrieval[qid]))

    return qid_docnos_nearest_neighbor_retrieval


#######################
# APPROACH 3 -  UNION #
#######################

def union_retrieval(qid_docnos_set_1, qid_docnos_set_2):
    qid_docnos_union_retrieval = {}

    # Combine the sets
    for key, value in qid_docnos_set_1.items():
        qid_docnos_union_retrieval[key] = value

    for key, value in qid_docnos_set_2.items():
        if key in qid_docnos_union_retrieval:
            qid_docnos_union_retrieval[key] = list(set(qid_docnos_union_retrieval[key] + value))
        else:
            qid_docnos_union_retrieval[key] = value

    return qid_docnos_union_retrieval


#########################
# WRITE RESULTS TO FILE #
#########################

def write_candidates(file_name, candidates, one_per_document):
    dataset = pt.get_dataset(DOCUMENT_DATASET_SOURCE_NAME_PYTERRIER)

    if one_per_document:
        # opd: add 15 known relevant and 5 known non-relevant passages for each query
        # maximum of one passage per document as known relevant or non-relevant passage
        with gzip.open(file_name, 'wt', encoding='UTF-8') as file:
            for query in dataset.irds_ref().queries_iter():

                qid = query.query_id
                # Only retrieve for the specified queries
                if QUERIES is not None and qid not in QUERIES:
                    continue
                if DOCUMENT_DATASET_SOURCE_NAME == 'disks45/nocr/trec-robust-2004' and qid == '672':
                    continue  # Skip query 672 as it has no relevant passages

                query_text = query.default_text()
                query_description = query.description if hasattr(query, 'description') else ""
                query_narrative = query.narrative if hasattr(query, 'narrative') else ""

                for target_docno in candidates[qid]:  # TODO: iterare over passages of docno
                    for target_passageno in target_docno_passagenos[target_docno]:
                        # Add 15 known relevant passages
                        for known_relevant_passageno in queries_best_passages_opd_cache[qid][:15]:
                            known_relevant_docno, _ = known_relevant_passageno.split(PASSAGE_ID_SEPARATOR)

                            file.write(json.dumps({
                                "qid": qid,
                                "query_text": query_text,
                                "query_description": query_description,
                                "query_narrative": query_narrative,
                                "source_dataset_id": DOCUMENT_DATASET_SOURCE_NAME,
                                "target_dataset_id": DOCUMENT_DATASET_TARGET_NAME,
                                "known_relevant_passage": {"docno": known_relevant_passageno,
                                                           "text": source_passages_text_cache[known_relevant_docno]
                                                           [known_relevant_passageno]},
                                "known_non_relevant_passage": "",  # False
                                "passage_to_judge": {"docno": target_passageno,
                                                     "text": target_passages_text_cache[target_docno][target_passageno]}
                            }) + '\n')
                        # 5 known non-relevant passages
                        for known_non_relevant_passageno in queries_worst_passages_opd_cache[qid][:5]:
                            known_non_relevant_docno, _ = known_non_relevant_passageno.split(PASSAGE_ID_SEPARATOR)

                            file.write(json.dumps({
                                "qid": qid,
                                "query_text": query_text,
                                "query_description": query_description,
                                "query_narrative": query_narrative,
                                "source_dataset_id": DOCUMENT_DATASET_SOURCE_NAME,
                                "target_dataset_id": DOCUMENT_DATASET_TARGET_NAME,
                                "known_relevant_passage": "",  # False
                                "known_non_relevant_passage": {"docno": known_non_relevant_passageno,
                                                               "text": source_passages_text_cache[known_non_relevant_docno]
                                                               [known_non_relevant_passageno]},
                                "passage_to_judge": {"docno": target_passageno,
                                                     "text": target_passages_text_cache[target_docno][target_passageno]}
                            }) + '\n')

    else:  # multiple passages of one document possible
        # Add 15 known relevant and 5 known non-relevant passages for each query
        # The top 15 (5) passages are allowed to be from the same source document
        with gzip.open(file_name, 'wt', encoding='UTF-8') as file:
            for query in dataset.irds_ref().queries_iter():

                qid = query.query_id
                # Only retrieve for the specified queries
                if QUERIES is not None and qid not in QUERIES:
                    continue
                if DOCUMENT_DATASET_SOURCE_NAME == 'disks45/nocr/trec-robust-2004' and qid == '672':
                    continue  # Skip query 672 as it has no relevant passages

                query_text = query.default_text()
                query_description = query.description if hasattr(query, 'description') else ""
                query_narrative = query.narrative if hasattr(query, 'narrative') else ""

                for target_docno in candidates[qid]:  # TODO: iterare over passages of docno
                    for target_passageno in target_docno_passagenos[target_docno]:
                        # Add 15 known relevant passages
                        for known_relevant_passageno in queries_best_passages_cache[qid][:15]:
                            known_relevant_docno, _ = known_relevant_passageno.split(PASSAGE_ID_SEPARATOR)

                            file.write(json.dumps({
                                "qid": qid,
                                "query_text": query_text,
                                "query_description": query_description,
                                "query_narrative": query_narrative,
                                "source_dataset_id": DOCUMENT_DATASET_SOURCE_NAME,
                                "target_dataset_id": DOCUMENT_DATASET_TARGET_NAME,
                                "known_relevant_passage": {"docno": known_relevant_passageno,
                                                           "text": source_passages_text_cache[known_relevant_docno]
                                                           [known_relevant_passageno]},
                                "known_non_relevant_passage": "",  # False
                                "passage_to_judge": {"docno": target_passageno,
                                                     "text": target_passages_text_cache[target_docno][target_passageno]}
                            }) + '\n')
                        # 5 known non-relevant passages
                        for known_non_relevant_passageno in queries_worst_passages_cache[qid][:5]:
                            known_non_relevant_docno, _ = known_non_relevant_passageno.split(PASSAGE_ID_SEPARATOR)

                            file.write(json.dumps({
                                "qid": qid,
                                "query_text": query_text,
                                "query_description": query_description,
                                "query_narrative": query_narrative,
                                "source_dataset_id": DOCUMENT_DATASET_SOURCE_NAME,
                                "target_dataset_id": DOCUMENT_DATASET_TARGET_NAME,
                                "known_relevant_passage": "",  # False
                                "known_non_relevant_passage": {"docno": known_non_relevant_passageno,
                                                               "text": source_passages_text_cache[known_non_relevant_docno]
                                                               [known_non_relevant_passageno]},
                                "passage_to_judge": {"docno": target_passageno,
                                                     "text": target_passages_text_cache[target_docno][target_passageno]}
                            }) + '\n')


if __name__ == '__main__':

    # Initialize the chunker
    chunker = PassageChunker()

    # Retrieve the document numbers for each approach
    # Maximum of 321 requests per query (16050 requests for 50 queries)
    docnos_naive = naive_retrieval(num_retrieval_docs=1000)  # 1 request/query

    docnos_nn_10 = nearest_neighbor_retrieval(
        num_top_passages=10, num_retrieval_docs=20, one_per_document=False)  # 10 requests/query
    docnos_nn_50 = nearest_neighbor_retrieval(
        num_top_passages=50, num_retrieval_docs=20, one_per_document=False)  # 50 requests/query
    docnos_nn_100 = nearest_neighbor_retrieval(
        num_top_passages=100, num_retrieval_docs=20, one_per_document=False)  # 100 requests/query

    docnos_nn_10_opd = nearest_neighbor_retrieval(
        num_top_passages=10, num_retrieval_docs=20, one_per_document=True)  # 10 requests/query
    docnos_nn_50_opd = nearest_neighbor_retrieval(
        num_top_passages=50, num_retrieval_docs=20, one_per_document=True)  # 50 requests/query
    docnos_nn_100_opd = nearest_neighbor_retrieval(
        num_top_passages=100, num_retrieval_docs=20, one_per_document=True)  # 100 requests/query

    docnos_union_10 = union_retrieval(docnos_naive, docnos_nn_10)  # 0 requests/query
    docnos_union_50 = union_retrieval(docnos_naive, docnos_nn_50)  # 0 requests/query
    docnos_union_100 = union_retrieval(docnos_naive, docnos_nn_100)  # 0 requests/query

    docnos_union_10_opd = union_retrieval(docnos_naive, docnos_nn_10_opd)  # 0 requests/query
    docnos_union_50_opd = union_retrieval(docnos_naive, docnos_nn_50_opd)  # 0 requests/query
    docnos_union_100_opd = union_retrieval(docnos_naive, docnos_nn_100_opd)  # 0 requests/query

    docnos_dicts = [
        docnos_union_10, docnos_union_50, docnos_union_100,
        docnos_union_10_opd, docnos_union_50_opd, docnos_union_100_opd
    ]

    # Flatten all docid lists and remove duplicates using set
    target_qid_docids = list(set(docid for docnos in docnos_dicts for docids in docnos.values() for docid in docids))
    print("Chunking target documents")
    chunker.chunk_target_documents(target_qid_docids, batch_size=2000)

    # File names
    naive_file_name = os.path.join(CANDIDATES_PATH, 'naive.jsonl.gz')
    naive_opd_file_name = os.path.join(CANDIDATES_PATH, 'naive_opd.jsonl.gz')

    nn_10_file_name = os.path.join(CANDIDATES_PATH, 'nearest_neighbor_10.jsonl.gz')
    nn_50_file_name = os.path.join(CANDIDATES_PATH, 'nearest_neighbor_50.jsonl.gz')
    nn_100_file_name = os.path.join(CANDIDATES_PATH, 'nearest_neighbor_100.jsonl.gz')

    nn_10_opd_file_name = os.path.join(CANDIDATES_PATH, 'nearest_neighbor_10_opd.jsonl.gz')
    nn_50_opd_file_name = os.path.join(CANDIDATES_PATH, 'nearest_neighbor_50_opd.jsonl.gz')
    nn_100_opd_file_name = os.path.join(CANDIDATES_PATH, 'nearest_neighbor_100_opd.jsonl.gz')

    union_10_file_name = os.path.join(CANDIDATES_PATH, 'union_10.jsonl.gz')
    union_50_file_name = os.path.join(CANDIDATES_PATH, 'union_50.jsonl.gz')
    union_100_file_name = os.path.join(CANDIDATES_PATH, 'union_100.jsonl.gz')

    union_10_opd_file_name = os.path.join(CANDIDATES_PATH, 'union_10_opd.jsonl.gz')
    union_50_opd_file_name = os.path.join(CANDIDATES_PATH, 'union_50_opd.jsonl.gz')
    union_100_opd_file_name = os.path.join(CANDIDATES_PATH, 'union_100_opd.jsonl.gz')

    # Write to file
    print("Writing naive candidates to file")
    write_candidates(naive_file_name, docnos_naive, one_per_document=False)
    write_candidates(naive_opd_file_name, docnos_naive, one_per_document=True)

    print("Writing nearest neighbor candidates to file")
    write_candidates(nn_10_file_name, docnos_nn_10, one_per_document=False)
    write_candidates(nn_50_file_name, docnos_nn_50, one_per_document=False)
    write_candidates(nn_100_file_name, docnos_nn_100, one_per_document=False)

    write_candidates(nn_10_opd_file_name, docnos_nn_10_opd, one_per_document=True)
    write_candidates(nn_50_opd_file_name, docnos_nn_50_opd, one_per_document=True)
    write_candidates(nn_100_opd_file_name, docnos_nn_100_opd, one_per_document=True)

    print("Writing union candidates to file")
    write_candidates(union_10_file_name, docnos_union_10, one_per_document=False)
    write_candidates(union_50_file_name, docnos_union_50, one_per_document=False)
    write_candidates(union_100_file_name, docnos_union_100, one_per_document=False)

    write_candidates(union_10_opd_file_name, docnos_union_10_opd, one_per_document=True)
    write_candidates(union_50_opd_file_name, docnos_union_50_opd, one_per_document=True)
    write_candidates(union_100_opd_file_name, docnos_union_100_opd, one_per_document=True)
