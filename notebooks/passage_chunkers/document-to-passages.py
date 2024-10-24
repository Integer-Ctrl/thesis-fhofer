import ir_datasets
import spacy
import pandas as pd
from tqdm import tqdm
from spacy_passage_chunker import SpacyPassageChunker

dataset = ir_datasets.load("argsme/2020-04-01")

'''
for doc in dataset.docs_iter():
    doc -> namedtuple<https://ir-datasets.com/argsme.html#argsme/2020-04-01>
'''


def check_for_ducplicates(num):
    counter = {}

    for doc in dataset.docs_iter()[:num]:
        if doc.source_url in counter:
            counter[doc.source_url] += 1
        else:
            counter[doc.source_url] = 1

    print(counter, "\n")


def static_document_segmentation(max_words=250, max_sentences=5):
    nlp = spacy.load("en_core_web_sm")
    chunked_docs = []

    for doc in tqdm(dataset.docs_iter()[:1000], desc="Processing documents", unit="doc"):
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
            "id": doc.source_id,
            "url": doc.source_url,
            "title": doc.source_title,
            "contents": segments
        })

    return chunked_docs


def dynamic_document_segmentation():
    chunker = SpacyPassageChunker()

    formatted_docs = []
    for doc in dataset.docs_iter()[:1000]:
        formatted_docs.append({
            "id": doc.source_id,
            "url": doc.source_url,
            "title": doc.source_title,
            "contents": doc.source_text
        })

    chunked_docs = chunker.process_batch(formatted_docs)

    return chunked_docs


# check_for_ducplicates(100) - contains many duplicates


# example usage of static_document_segmentation - slow
# chunked_docs = static_document_segmentation(300, 20)
# print(chunked_docs[0])


# example usage of dynamic_document_segmentation - fast
chunked_documents = dynamic_document_segmentation()
print(chunked_documents[0])
