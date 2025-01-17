import ir_datasets
from tqdm import tqdm
from spacy_passage_chunker import SpacyPassageChunker
import json
import os
import gzip


# Load the configuration settings
pwd = os.path.dirname(os.path.abspath(__file__))


def load_config(filename=pwd + "/../config.json"):
    with open(filename, "r") as f:
        config = json.load(f)
    return config


# Get the configuration settings
config = load_config()
PASSAGE_ID_SEPARATOR = config['PASSAGE_ID_SEPARATOR']

DOCUMENT_DATASET_SOURCE_NAME = config['DOCUMENT_DATASET_SOURCE_NAME']
DOCUMENT_DATASET_SOURCE_NAME_PYTHON_API = config['DOCUMENT_DATASET_SOURCE_NAME_PYTHON_API']

SOURCE_PATH = os.path.join(config['DATA_PATH'], DOCUMENT_DATASET_SOURCE_NAME)
PASSAGE_DATASET_SOURCE_PATH = os.path.join(SOURCE_PATH, config['PASSAGE_DATASET_SOURCE_PATH'])


# Class to chunk documents into passages
class PassageChunker:

    def __init__(self, ir_dataset):
        self.dataset = ir_dataset
        self.docstore = self.dataset.docs_store()

    def dynamic_document_segmentation(self, path, docs_to_chunk, batch_size=1000):
        # Initialize the passage chunker
        chunker = SpacyPassageChunker()

        BATCH_SIZE = batch_size
        batch = []
        known_doc_ids = set()
        chunked_docs_count = 0

        docs_dict = self.docstore.get_many(docs_to_chunk)
        print(f"Loaded {len(docs_dict)} documents from {len(docs_to_chunk)} docs to chunk")

        # Open the output file in append mode
        with gzip.open(path, 'wt', encoding='UTF-8') as file:

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
                    # Chunk the batch of documents
                    chunked_docs_count += len(batch)
                    print(f"Chunking documents: {chunked_docs_count}")
                    chunked_batch = chunker.process_batch(batch)

                    for chunked_doc in chunked_batch:
                        # Write each passage as a JSON object
                        for passage in chunked_doc['contents']:
                            passage_id = chunked_doc['docno'] + PASSAGE_ID_SEPARATOR + str(passage['id'])
                            file.write(
                                (json.dumps({"docno": passage_id, "text": passage['body']}) + '\n'))

                    # Reset the batch after saving
                    batch = []

            # Process and save any remaining documents in the batch
            if batch:
                chunked_docs_count += len(batch)
                chunked_batch = chunker.process_batch(batch)

                for chunked_doc in chunked_batch:
                    # Write each passage as a JSON object
                    for passage in chunked_doc['contents']:
                        passage_id = chunked_doc['docno'] + PASSAGE_ID_SEPARATOR + str(passage['id'])
                        file.write(
                            (json.dumps({"docno": passage_id, "text": passage['body']}) + '\n'))

        print(f"Processed and saved {chunked_docs_count} documents to {PASSAGE_DATASET_SOURCE_PATH}")


# Get list of doc ids that should be chunked
# For each QID, chunk 50 non relevant documents with a label <= 0
# For each QID, chunk 50 relevant documents for each label > 0
# If there are less than 50 documents for a label, chunk for each label as much as the smallest label count
def get_docs_to_chunk(dataset):
    dict = {}

    for qrel in dataset.qrels_iter():
        qid = qrel.query_id
        doc_id = qrel.doc_id
        label = qrel.relevance

        if qid not in dict:
            dict[qid] = {}

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
        min_label_count = min([[len(count)] for count in dict[qid].values()])
        min_label_count = min(min_label_count[0], 50)

        for label in dict[qid]:
            dict[qid][label] = dict[qid][label][:min_label_count]

        print(f"QID: {qid} has {len(dict[qid].keys())} labels with {min_label_count} documents each")

    # Flatten the dictionary
    doc_ids = []
    for qid in dict:
        for label in dict[qid]:
            doc_ids += dict[qid][label]

    return set(doc_ids)


# Chunk source dataset and save to file
dataset = ir_datasets.load(DOCUMENT_DATASET_SOURCE_NAME_PYTHON_API)

qid_docids = get_docs_to_chunk(dataset)

chunker = PassageChunker(dataset)
chunker.dynamic_document_segmentation(PASSAGE_DATASET_SOURCE_PATH, qid_docids, batch_size=200)
