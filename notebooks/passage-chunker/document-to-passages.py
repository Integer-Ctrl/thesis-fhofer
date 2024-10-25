import ir_datasets
import spacy
import pandas as pd
from tqdm import tqdm
from spacy_passage_chunker import SpacyPassageChunker
import json
import zipfile
import os

'''
for doc in dataset.docs_iter():
    doc -> namedtuple<https://ir-datasets.com/argsme.html#argsme/2020-04-01>
'''


class TestPassageChunker:

    def __init__(self, ir_dataset='argsme/2020-04-01'):
        self.dataset = ir_datasets.load(ir_dataset)

        def check_for_ducplicates(self, num):
            counter = {}

            for doc in self.dataset.docs_iter()[:num]:
                if doc.source_url in counter:
                    counter[doc.source_url] += 1
                else:
                    counter[doc.source_url] = 1

            print(counter, '\n')

    def static_document_segmentation(self, max_words=250, max_sentences=5):
        nlp = spacy.load('en_core_web_sm')
        chunked_docs = []

        for doc in tqdm(self.dataset.docs_iter()[:1000], desc='Processing documents', unit='doc'):
            spacy_doc = nlp(doc.source_text)
            segments = []
            current_segment = []
            current_word_count = 0
            current_sentence_count = 0

            for sent in spacy_doc.sents:
                # Get word count of the current sentence
                sent_word_count = len(sent.text.split())

                # Check if adding this sentence exceeds word or sentence limits
                if (current_word_count + sent_word_count > max_words) or (current_sentence_count + 1 > max_sentences):
                    # If limits exceeded, save the current segment
                    segments.append(' '.join(current_segment))
                    # Reset counters for new segment
                    current_segment = []
                    current_word_count = 0
                    current_sentence_count = 0

                # Add sentence to current segment
                current_segment.append(sent.text)
                current_word_count += sent_word_count
                current_sentence_count += 1

            # Add any leftover sentences as the last segment
            if current_segment:
                segments.append(' '.join(current_segment))

            chunked_docs.append({
                'id': doc.source_id,
                'url': doc.source_url,
                'title': doc.source_title,
                'contents': segments
            })

        return chunked_docs


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
            f_out.write('{"documents": [\n{"id": "", "url": "", "title": "", "contents": [{"body": "", "id": ""}]}')

            for doc in tqdm(self.dataset.docs_iter()[:100], desc='Chunking and saving documents', unit='doc'):
                # Format the document
                formatted_doc = {
                    'id': doc.source_id,
                    'url': doc.source_url,
                    'title': doc.source_title,
                    'contents': doc.source_text
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


# check_for_ducplicates(100) - contains many duplicates


# example usage of static_document_segmentation - slow
# chunked_docs = static_document_segmentation(300, 20)
# print(chunked_docs[0])


# example usage of dynamic_document_segmentation - fast
chunker = PassageChunker()
chunker.dynamic_document_segmentation(file_name='chunked-docs.json', batch_size=10)
