import json
import os
import pyterrier as pt
import ir_datasets
import gzip
from tqdm import tqdm

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


check_scores_smaller_zero()
