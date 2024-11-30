import ir_datasets
from tqdm import tqdm
from spacy_passage_chunker import SpacyPassageChunker
import json
import os
import gzip


# Load the configuration settings
def load_config(filename="../config.json"):
    with open(filename, "r") as f:
        config = json.load(f)
    return config


# Get the configuration settings
config = load_config()

DOCUMENT_DATASET_OLD_NAME = config['DOCUMENT_DATASET_OLD_NAME']
DOCUMENT_DATASET_OLD_NAME_PYTHON_API = config['DOCUMENT_DATASET_OLD_NAME_PYTHON_API']

DATA_PATH = os.path.join(config['DATA_PATH'], DOCUMENT_DATASET_OLD_NAME)

PASSAGE_DATASET_PATH = os.path.join(DATA_PATH, config['PASSAGE_DATASET_PATH'])
PASSAGE_ID_SEPARATOR = config['PASSAGE_ID_SEPARATOR']


class PassageChunker:

    def __init__(self, ir_dataset):
        self.dataset = ir_datasets.load(ir_dataset)

    def dynamic_document_segmentation(self, batch_size=1000):
        # Initialize the passage chunker
        chunker = SpacyPassageChunker()

        BATCH_SIZE = batch_size
        batch = []
        doc_count = 0
        known_doc_ids = set()

        # Open the output file in append mode
        with gzip.open(PASSAGE_DATASET_PATH, 'wt', encoding='UTF-8') as file:

            for doc in tqdm(self.dataset.docs_iter(), desc='Chunking and saving documents', unit='doc'):
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
                    chunked_batch = chunker.process_batch(batch)

                    for chunked_doc in chunked_batch:
                        # Write each passage as a JSON object
                        for passage in chunked_doc['contents']:
                            passage_id = chunked_doc['docno'] + PASSAGE_ID_SEPARATOR + str(passage['id'])
                            file.write(
                                (json.dumps({"docno": passage_id, "text": passage['body']}) + '\n'))

                    # Reset the batch after saving
                    batch = []

                    # Keep track of how many documents are processed
                    doc_count += BATCH_SIZE

            # Process and save any remaining documents in the batch
            if batch:
                chunked_batch = chunker.process_batch(batch)

                for chunked_doc in chunked_batch:
                    # Write each passage as a JSON object
                    for passage in chunked_doc['contents']:
                        passage_id = chunked_doc['docno'] + PASSAGE_ID_SEPARATOR + str(passage['id'])
                        file.write(
                            (json.dumps({"docno": passage_id, "text": passage['body']}) + '\n'))

                doc_count += len(batch)

        print(f"Processed and saved {doc_count} documents to {PASSAGE_DATASET_PATH}")


chunker = PassageChunker(DOCUMENT_DATASET_OLD_NAME_PYTHON_API)
chunker.dynamic_document_segmentation(batch_size=4000)
