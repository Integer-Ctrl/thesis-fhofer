import json
import os
import pyterrier as pt
from tqdm import tqdm
import gzip
from trectools import TrecQrel, TrecRun, TrecEval
import sys
import time

# Load the configuration settings
pwd = os.path.dirname(os.path.abspath(__file__))


def load_config(filename=pwd + "/../config.json"):
    with open(filename, "r") as f:
        config = json.load(f)
    return config


# Get the configuration settings
config = load_config()

DOCUMENT_DATASET_SOURCE_NAME = config['DOCUMENT_DATASET_SOURCE_NAME']
DOCUMENT_DATASET_SOURCE_NAME_PYTERRIER = config['DOCUMENT_DATASET_SOURCE_NAME_PYTERRIER']

SOURCE_PATH = os.path.join(config['DATA_PATH'], DOCUMENT_DATASET_SOURCE_NAME)
DOCUMENT_DATASET_SOURCE_INDEX_PATH = os.path.join(SOURCE_PATH, config['DOCUMENT_DATASET_SOURCE_INDEX_PATH'])

PASSAGE_ID_SEPARATOR = config['PASSAGE_ID_SEPARATOR']
CHATNOIR_RETRIEVAL = config['CHATNOIR_RETRIEVAL']

# Initialize PyTerrier and Tokenizer
if not pt.java.started():
    pt.java.init()
tokeniser = pt.java.autoclass('org.terrier.indexing.tokenisation.Tokeniser').getTokeniser()


# Document yield function for indexing without duplicates
def yield_docs(dataset):
    known_docnos = set()
    for i in dataset.irds_ref().docs_iter():
        if i.doc_id not in known_docnos:
            known_docnos.add(i.doc_id)
            yield {'docno': i.doc_id, 'text': i.default_text()}


# Build index if not clueweb
if CHATNOIR_RETRIEVAL:
    print("Chatnoir dataset, no need to build index")
else:
    # Index source document dataset
    if not os.path.exists(DOCUMENT_DATASET_SOURCE_INDEX_PATH):
        print(f"Indexing dataset {DOCUMENT_DATASET_SOURCE_NAME}")
        dataset = pt.get_dataset(DOCUMENT_DATASET_SOURCE_NAME_PYTERRIER)
        indexer = pt.IterDictIndexer(DOCUMENT_DATASET_SOURCE_INDEX_PATH)
        index_ref = indexer.index(yield_docs(dataset),
                                  meta={'docno': 50, 'text': 20000})
    else:
        print("Index already exists")
        index_ref = pt.IndexRef.of(DOCUMENT_DATASET_SOURCE_INDEX_PATH + '/data.properties')

    dataset_index = pt.IndexFactory.of(index_ref)
