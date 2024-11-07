import ir_datasets
from tqdm import tqdm
from spacy_passage_chunker import SpacyPassageChunker
import json
import os
import gzip

'''
for doc in dataset.docs_iter():
    doc -> namedtuple<https://ir-datasets.com/argsme.html#argsme/2020-04-01>
'''

# DATASET_NAME = 'argsme/2020-04-01/touche-2020-task-1'  # Python API dataset name
DATASET_NAME = 'argsme/2020-04-01/touche-2021-task-1'  # Python API dataset name
DATASET_PATH = '../data'
SEPARATOR = '___'


class PassageChunker:

    def __init__(self, ir_dataset):
        self.dataset = ir_datasets.load(ir_dataset)

    def dynamic_document_segmentation(self, file_name, batch_size=1000):
        # Initialize the passage chunker
        chunker = SpacyPassageChunker()

        BATCH_SIZE = batch_size
        batch = []
        doc_count = 0

        path = os.path.join(DATASET_PATH, file_name)

        # Open the output file in append mode
        with gzip.open(path, 'wt', encoding='UTF-8') as f_out:

            for doc in tqdm(self.dataset.docs_iter(), desc='Chunking and saving documents', unit='doc'):
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
                            passage_id = chunked_doc['docno'] + SEPARATOR + str(passage['id'])
                            f_out.write(
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
                        passage_id = chunked_doc['docno'] + SEPARATOR + str(passage['id'])
                        f_out.write(
                            (json.dumps({"docno": passage_id, "text": passage['body']}) + '\n'))

                doc_count += len(batch)

        print(f"Processed and saved {doc_count} documents to {file_name}")


chunker = PassageChunker(DATASET_NAME)
chunker.dynamic_document_segmentation(file_name=DATASET_NAME + '/passage-dataset/passages.jsonl.gz', batch_size=4000)
