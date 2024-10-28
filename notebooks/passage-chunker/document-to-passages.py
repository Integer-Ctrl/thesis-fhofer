import ir_datasets
from tqdm import tqdm
from spacy_passage_chunker import SpacyPassageChunker
import json

'''
for doc in dataset.docs_iter():
    doc -> namedtuple<https://ir-datasets.com/argsme.html#argsme/2020-04-01>
'''


class PassageChunker:

    def __init__(self, ir_dataset='argsme/2020-04-01'):
        self.dataset = ir_datasets.load(ir_dataset)

    def dynamic_document_segmentation(self, file_name='chunked-docs.json', batch_size=1000):
        # Initialize the passage chunker
        chunker = SpacyPassageChunker()

        BATCH_SIZE = batch_size
        batch = []
        doc_count = 0

        # Open the output file in append mode
        with open(file_name, 'a') as f_out:
            f_out.write('{"documents": [\n{"docno": "", "url": "", "title": "", "contents": [{"body": "", "id": ""}]}')

            for doc in tqdm(self.dataset.docs_iter()[:10], desc='Chunking and saving documents', unit='doc'):
                # Format the document
                formatted_doc = {
                    'docno': doc.doc_id,
                    'url': doc.source_url,
                    'title': doc.source_title,
                    'contents': doc.default_text()
                }

                # Add the document to the current batch
                batch.append(formatted_doc)

                # If the batch reaches the specified batch size, process and save it
                if len(batch) >= BATCH_SIZE:
                    # Chunk the batch of documents
                    chunked_batch = chunker.process_batch(batch)

                    # Write each chunked document to the file in JSONL format
                    for chunked_doc in chunked_batch:
                        f_out.write(',\n' + json.dumps(chunked_doc))  # Write each chunk as a JSON object

                    # Reset the batch after saving
                    batch = []

                    # Keep track of how many documents are processed
                    doc_count += BATCH_SIZE

            # Process and save any remaining documents in the batch
            if batch:
                chunked_batch = chunker.process_batch(batch)

                for chunked_doc in chunked_batch:
                    f_out.write(',\n' + json.dumps(chunked_doc))

                doc_count += len(batch)

            # Write the closing of the JSON array and object
            f_out.write('\n]}')

        print(f"Processed and saved {doc_count} documents to {file_name}")


chunker = PassageChunker()
chunker.dynamic_document_segmentation(file_name='chunked-docs-small.json', batch_size=4000)
