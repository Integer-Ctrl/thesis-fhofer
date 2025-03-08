from tqdm import tqdm
import gzip
import json
import os
import ir_datasets
import pyterrier as pt
from glob import glob
import pandas as pd
from spacy_passage_chunker import SpacyPassageChunker
import time
from ir_datasets_clueweb22 import register

# Register the ClueWeb22/b dataset
register()

def candidate_retrieval():

    def load_config(filename="/mnt/ceph/storage/data-tmp/current/ho62zoq/thesis-fhofer/code/src/config.json"):
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
    # PASSAGE_DATASET_SOURCE_SCORE_REL_PATH = os.path.join(SOURCE_PATH, config['PASSAGE_DATASET_SOURCE_SCORE_REL_PATH'])
    PASSAGE_DATASET_SOURCE_SCORE_PATH = os.path.join(SOURCE_PATH, config['PASSAGE_DATASET_SOURCE_SCORE_AQ_PATH'])
    FILE_PATTERN = os.path.join(PASSAGE_DATASET_SOURCE_SCORE_PATH, "qid_*.jsonl.gz")

    if CHATNOIR_RETRIEVAL:
        CANDIDATES_PATH = os.path.join(TARGET_PATH, config['CANDIDATE_CHATNOIR_PATH'])
    else:
        CANDIDATES_PATH = os.path.join(TARGET_PATH, config['CANDIDATES_LOCAL_PATH'])

    RANK_CORRELATION_SCORE_AVG_PATH = os.path.join(SOURCE_PATH, config['RANK_CORRELATION_SCORE_AVG_PATH'])

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

    # Get type of metric with highest rank correlation (decending order in file)
    best_scoring_metric_retriever = None
    with gzip.open(RANK_CORRELATION_SCORE_AVG_PATH, 'rt', encoding='UTF-8') as file:
        for line in file:  # already decending sorted
            data = json.loads(line)  # return only best scoring method
            _, _, _, retriever, metric = data['eval---aggr---tra---retriever---metric'].split(KEY_SEPARATOR)

            best_scoring_metric_retriever = metric + '_' + retriever
            print(f"Best scoring metric retriever: {best_scoring_metric_retriever}")
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
                passages_score_cache[qid][docno] = {}
                passages_score_cache[qid][docno]['score'] = data[best_scoring_metric_retriever]  # assigned passage score
                passages_score_cache[qid][docno]['label'] = data['label']                        # actual label of retrieval task

    """
    Determine for all queries the best and worst passages
        - one approach limits the number of passages per document to one (opd)
        - the other approach has no limitation
        - ensure that best passages are from relevant documents and worst passages are from non-relevant documents
    """
    queries_best_passages_cache = {}  # multiple passages of one document possible
    queries_worst_passages_cache = {}  # multiple passages of one document possible
    queries_best_passages_opd_cache = {}  # opd = one per documnet, maximum of one passage per document
    queries_worst_passages_opd_cache = {}  # opd = one per documnet, maximum of one passage per document

    for qid, passageno_scores in passages_score_cache.items():
        # Parse docnos and sort by score
        docnos_best_passagenos = []
        docnos_worst_passagenos = []
        docnos_best_passagenos_opd = {}   # opd
        docnos_worst_passagenos_opd = {}  # opd

        for passageno, score_label in passageno_scores.items():
            # Extract docno by removing the suffix ___x
            docno, _ = passageno.split(PASSAGE_ID_SEPARATOR)
            score = score_label['score']
            label = score_label['label']

            # Relevant passages
            if label > 0:
                docnos_best_passagenos += [(passageno, score)]
                # Keep the highest-scoring passageno for each docno for opd approach
                if (docno not in docnos_best_passagenos_opd or score > docnos_best_passagenos_opd[docno][1]):
                    docnos_best_passagenos_opd[docno] = (passageno, score)

            # Non-relevant passages
            if label <= 0:
                docnos_worst_passagenos += [(passageno, score)]
                # Keep the lowest-scoring passageno for each docno for opd approach
                if (docno not in docnos_worst_passagenos_opd or score < docnos_worst_passagenos_opd[docno][1]):
                    docnos_worst_passagenos_opd[docno] = (passageno, score)

        # Sort by score descending
        queries_best_passages_cache[qid] = [item[0] for item in sorted(docnos_best_passagenos, key=lambda x: x[1], reverse=True)]
        # Sort by score ascending
        queries_worst_passages_cache[qid] = [item[0] for item in sorted(docnos_worst_passagenos, key=lambda x: x[1])]

        # opd: Extract highest-scored passagenos and sort them in descending order
        best_passagenos = [item[0] for item in sorted(docnos_best_passagenos_opd.values(), key=lambda x: x[1], reverse=True)]
        queries_best_passages_opd_cache[qid] = best_passagenos
        # opd: Extract lowest-scored passagenos and sort them in ascending order
        worst_passagenos = [item[0]for item in sorted(docnos_worst_passagenos_opd.values(), key=lambda x: x[1])]
        queries_worst_passages_opd_cache[qid] = worst_passagenos

    # Get for each query all relevant passages in dictionary format qid: [passageno]
    source_docno_passagenos = {}
    # Get all passages text in dictionary format docno: {passageno: text}
    source_passages_text_cache = {}

    with gzip.open(PASSAGE_DATASET_SOURCE_PATH, 'rt', encoding='UTF-8') as file:
        for line in file:
            line = json.loads(line)
            docno, passageno = line['docno'].split(PASSAGE_ID_SEPARATOR)
            if docno not in source_docno_passagenos:
                source_docno_passagenos[docno] = []
                source_passages_text_cache[docno] = {}
            source_docno_passagenos[docno] += [line['docno']]
            source_passages_text_cache[docno][line['docno']] = line['text']


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
                    if DOCUMENT_DATASET_SOURCE_NAME == 'disks45/nocr/trec-robust-2004' and qid == '306':
                        print("Writing query 306")

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
                    if DOCUMENT_DATASET_SOURCE_NAME == 'disks45/nocr/trec-robust-2004' and qid == '306':
                        print("Writing query 306")

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

    ############################
    #           MAIN           #
    ############################

    # Initialize the chunker
    chunker = PassageChunker()

    # Load docnos for evaluation
    docnos_eval = {}
    with open('/mnt/ceph/storage/data-tmp/current/ho62zoq/data/clueweb22-transfer/judgment-pool-10.json', 'r') as file:
        data = json.load(file)
        for qid in QUERIES:
            docnos_eval[qid] = list(set(data[qid]))

    # Flatten all docid lists and remove duplicates using set
    target_qid_docids = list(set([docid for docids in docnos_eval.values() for docid in docids]))
    print("Chunking target documents")
    chunker.chunk_target_documents(target_qid_docids, batch_size=2000)

    eval_file_name = os.path.join(CANDIDATES_PATH, 'eval_candidates.jsonl.gz')
    write_candidates(eval_file_name, docnos_eval, one_per_document=True)


if __name__ == '__main__':

    candidate_retrieval()
