import json
import os
import pyterrier as pt
import ir_datasets
import gzip
from tqdm import tqdm
from collections import Counter

DATASET_NAME = 'irds:argsme/2020-04-01/touche-2021-task-1'  # PyTerrier dataset name
PASSAGE_PATH = 'data/' + DATASET_NAME.replace('irds:', '') + '/passage-dataset/passages.jsonl.gz'
PASSAGE_SCORES_PATH = 'data/' + DATASET_NAME.replace('irds:', '') + '/passage-dataset/passage-scores.jsonl.gz'
PASSAGE_TO_DOCUMENT_SCORES_PATH = 'data/' + \
    DATASET_NAME.replace('irds:', '') + '/document-dataset/passages-to-document/correlation-scores.jsonl.gz'

AGGREGATION_METHODS = ['mean', 'max', 'min']
TRANSFORMATION_METHODS = ['id', 'log', 'binned']
EVALUATION_METHODS = ['pearson_pd', 'pearson_scipy', 'kendall_pd', 'kendall_scipy', 'spearman_pd', 'spearman_scipy']
METRICS = ['p10_bm25', 'p10_bm25_wod', 'p10_tfidf', 'p10_tfidf_wod',
           'ndcg10_bm25', 'ndcg10_bm25_wod', 'ndcg10_tfidf', 'ndcg10_tfidf_wod']


# Load the configuration settings
def load_config(filename="./config.json"):
    with open(filename, "r") as f:
        config = json.load(f)
    return config


# Get the configuration settings
config = load_config()


def read_json():
    with open('data/chunked-docs.json') as f:
        data = json.load(f)

    for doc in data['documents']:
        if doc['docno'] == 'S2db48a61-A430a7cb1':
            print('ID:', doc['docno'])
            print(doc['contents'], '\n')


def docs_pyterrier():
    dataset = pt.get_dataset('irds:argsme/2020-04-01')
    for doc in dataset.get_corpus_iter():
        print(doc.keys(), '\n')
        print(doc['docno'], '\n')
        print(doc['conclusion'], '\n')
        print(doc['topic'], '\n')
        print(doc['premises'], '\n')
        break


def docs_ir_dataset():
    dataset = ir_datasets.load('argsme/2020-04-01')
    for doc in dataset.docs_iter()[:1]:
        print(doc, '\n')
        break


def ir_read_qrels():
    dataset = ir_datasets.load("argsme/2020-04-01/touche-2020-task-1")
    for qrel in dataset.qrels_iter():
        print(qrel, '\n')


def pt_read_qrels():
    dataset = pt.get_dataset("irds:argsme/2020-04-01/touche-2020-task-1")
    count = 0
    for index, row in dataset.get_qrels().iterrows():
        count += 1
    print(count)


def check_scores_smaller_zero():
    docno_qid_passages_scores_cache = {}
    with gzip.open(PASSAGE_SCORES_PATH, 'rt', encoding='UTF-8') as file:
        for line in tqdm(file, desc='Caching passage scores', unit='passage'):
            line = json.loads(line)
            docno, passageno = line['docno'].split('___')
            qid = line['qid']
            if docno not in docno_qid_passages_scores_cache:
                docno_qid_passages_scores_cache[docno] = {}
            if qid not in docno_qid_passages_scores_cache[docno]:
                docno_qid_passages_scores_cache[docno][qid] = []
            docno_qid_passages_scores_cache[docno][qid] += [line]

    for docno, qid_passages_scores in docno_qid_passages_scores_cache.items():
        print(qid_passages_scores)
        for qid, passages_scores in qid_passages_scores.items():
            for metric in METRICS:
                scores = [passage[metric] for passage in passages_scores]
                print(scores)
                if any(score < 0 for score in scores):
                    print(docno, qid, metric, scores)

    print('Done')


# checks if there is an qid for which only one label value is present
def check_qrels_for_single_label():
    qrels_cache = {}
    dataset = pt.get_dataset('irds:argsme/2020-04-01/touche-2021-task-1')
    qrels = dataset.get_qrels()
    for index, row in tqdm(qrels.iterrows(), desc='Caching qrels', unit='qrel'):
        if row['label'] > 0:
            if row['qid'] not in qrels_cache:
                qrels_cache[row['qid']] = []
            qrels_cache[row['qid']] += [row['label']]

    for qid, labels in qrels_cache.items():
        if len(set(labels)) == 1:
            print(qid, labels)

    print('Done')


# check if passages contain dublicates
def check_passages_ducplicate():
    DOCUMENT_DATASET_OLD_NAME = config['DOCUMENT_DATASET_OLD_NAME']
    DATA_PATH = os.path.join(config['DATA_PATH'], DOCUMENT_DATASET_OLD_NAME)
    PASSAGE_DATASET_OLD_PATH = os.path.join(DATA_PATH, config['PASSAGE_DATASET_OLD_PATH'])

    passagesnos_cnt = Counter()
    passagesnos_list = []
    with gzip.open(PASSAGE_DATASET_OLD_PATH, 'rt', encoding='UTF-8') as file:
        for line in tqdm(file, desc='Caching passages', unit='passage'):
            line = json.loads(line)
            passagesnos_cnt[line['docno']] += 1
            passagesnos_list.append(line['docno'])

    print(len(passagesnos_list) != len(set(passagesnos_list)))
    print(passagesnos_cnt.most_common(10))


# check if dataset contain dublicates
def check_dataset_ducplicate():
    DOCUMENT_DATASET_OLD_NAME_PYTHON_API = config['DOCUMENT_DATASET_OLD_NAME_PYTHON_API']

    dataset = ir_datasets.load(DOCUMENT_DATASET_OLD_NAME_PYTHON_API)
    docnos_cnt = Counter()
    docnos_list = []

    for doc in dataset.docs_iter():
        docnos_cnt[doc.doc_id] += 1
        docnos_list.append(doc.doc_id)

    print(len(docnos_list) != len(set(docnos_list)))
    print(docnos_cnt.most_common(10))
