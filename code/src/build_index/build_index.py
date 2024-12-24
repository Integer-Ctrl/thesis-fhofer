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

PT_RETRIEVERS = config['PT_RETRIEVERS']

DOCUMENT_DATASET_OLD_NAME = config['DOCUMENT_DATASET_OLD_NAME']
DOCUMENT_DATASET_OLD_NAME_PYTERRIER = config['DOCUMENT_DATASET_OLD_NAME_PYTERRIER']

OLD_PATH = os.path.join(config['DATA_PATH'], DOCUMENT_DATASET_OLD_NAME)

DOCUMENT_DATASET_OLD_INDEX_PATH = os.path.join(OLD_PATH, config['DOCUMENT_DATASET_OLD_INDEX_PATH'])

PASSAGE_DATASET_OLD_PATH = os.path.join(OLD_PATH, config['PASSAGE_DATASET_OLD_PATH'])

PASSAGE_DATASET_OLD_SCORE_AQ_PATH = os.path.join(OLD_PATH, config['PASSAGE_DATASET_OLD_SCORE_AQ_PATH'])
PASSAGE_DATASET_OLD_SCORE_REL_PATH = os.path.join(OLD_PATH, config['PASSAGE_DATASET_OLD_SCORE_REL_PATH'])

PASSAGE_ID_SEPARATOR = config['PASSAGE_ID_SEPARATOR']

# Initialize PyTerrier and Tokenizer
if not pt.java.started():
    pt.java.init()
tokeniser = pt.java.autoclass('org.terrier.indexing.tokenisation.Tokeniser').getTokeniser()

dataset = pt.get_dataset(DOCUMENT_DATASET_OLD_NAME_PYTERRIER)


# Document yield function for indexing without duplicates
def yield_docs(dataset):
    known_docnos = set()
    for i in dataset.irds_ref().docs_iter():
        if i.doc_id not in known_docnos:
            known_docnos.add(i.doc_id)
            yield {'docno': i.doc_id, 'text': i.default_text()}


# Index dataset
if not os.path.exists(DOCUMENT_DATASET_OLD_INDEX_PATH):
    indexer = pt.IterDictIndexer(DOCUMENT_DATASET_OLD_INDEX_PATH)
    index_ref = indexer.index(yield_docs(dataset),
                              meta={'docno': 50, 'text': 20000})
else:
    index_ref = pt.IndexRef.of(DOCUMENT_DATASET_OLD_INDEX_PATH + '/data.properties')

dataset_index = pt.IndexFactory.of(index_ref)
