import json
import os
import pyterrier as pt
import ir_datasets
import gzip
from tqdm import tqdm
from collections import Counter
from glob import glob
from ir_datasets_clueweb22 import register

register()

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
pwd = os.path.dirname(os.path.abspath(__file__))


def load_config(filename=pwd + "/config.json"):
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
    for index, row in dataset.get_qrels(variant='relevance').iterrows():
        if row['label'] > 0:
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
    qrels = dataset.get_qrels(variant='relevance')
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
def check_passages_ducplicate(path):
    passagesnos_cnt = Counter()
    passagesnos_list = []
    with gzip.open(path, 'rt', encoding='UTF-8') as file:
        for line in tqdm(file, desc='Caching passages', unit='passage'):
            line = json.loads(line)
            passagesnos_cnt[line['docno']] += 1
            passagesnos_list.append(line['docno'])

    print(len(passagesnos_list) != len(set(passagesnos_list)))
    print(passagesnos_cnt.most_common(10))


# check if dataset contain dublicates
def check_dataset_ducplicate():
    DOCUMENT_DATASET_SOURCE_NAME_PYTHON_API = config['DOCUMENT_DATASET_SOURCE_NAME_PYTHON_API']

    dataset = ir_datasets.load(DOCUMENT_DATASET_SOURCE_NAME_PYTHON_API)
    docnos_cnt = Counter()
    docnos_list = []

    for doc in dataset.docs_iter():
        docnos_cnt[doc.doc_id] += 1
        docnos_list.append(doc.doc_id)

    print(len(docnos_list) != len(set(docnos_list)))
    print(docnos_cnt.most_common(10))


def trim_string(string, limit):
    return string[:limit]


def check_alloc_mem():
    cache = {}
    known_doc_ids = set()

    dataset = ir_datasets.load('msmarco-document/trec-dl-2019/judged')

    for doc in dataset.docs_iter():
        if doc.doc_id in known_doc_ids:
            continue
        known_doc_ids.add(doc.doc_id)
        cache[doc.doc_id] = trim_string(doc.default_text(), 2000000)

    return cache


qid_docnos_naive_retrieval = {1: ['doc1', 'doc2', 'doc3'], 2: ['doc4', 'doc5', 'doc6']}
qid_docnos_nearest_neighbor_retrieval = {1: ['doc1', 'doc8', 'doc9'], 2: ['doc10', 'doc11', 'doc12']}


def combine_dicts():

    qid_docnos_union_retrieval = {}

    def union_retrieval():

        for key, value in qid_docnos_naive_retrieval.items():
            qid_docnos_union_retrieval[key] = value

        for key, value in qid_docnos_nearest_neighbor_retrieval.items():
            if key in qid_docnos_union_retrieval:
                qid_docnos_union_retrieval[key] = list(set(qid_docnos_union_retrieval[key] + value))
            else:
                qid_docnos_union_retrieval[key] = value

        return qid_docnos_union_retrieval

    print(union_retrieval())


def check_query_keys():
    dataset = pt.get_dataset('irds:argsme/2020-04-01/touche-2021-task-1')
    for query in dataset.get_topics():
        break


def check_correlation_dublicates():

    DOCUMENT_DATASET_SOURCE_NAME = config['DOCUMENT_DATASET_SOURCE_NAME']
    SOURCE_PATH = os.path.join(config['DATA_PATH'], DOCUMENT_DATASET_SOURCE_NAME)
    KEY_SEPARATOR = config['KEY_SEPARATOR']
    RANK_CORRELATION_SCORE_PQ_AQ_PATH = os.path.join(
        SOURCE_PATH, config['RANK_CORRELATION_SCORE_PQ_AQ_PATH'])
    FILE_PATTERN = os.path.join(RANK_CORRELATION_SCORE_PQ_AQ_PATH, "job_*.jsonl.gz")
    PT_RETRIEVERS = config['PT_RETRIEVERS']

    def get_key(list):
        return KEY_SEPARATOR.join(list)

    files = 0
    correlation_scores_eva_ret = {}
    for file_path in glob.glob(FILE_PATTERN):
        with gzip.open(file_path, 'rt', encoding='UTF-8') as file:
            files += 1
            for line in file:
                line = json.loads(line)
                agr_met = line['aggregation_method']
                tra_met = line['transformation_method']
                eva_met = line['evaluation_method']
                for pt_retriever in PT_RETRIEVERS:
                    if pt_retriever in line['metric']:  # eg p10_BM25
                        retriever = pt_retriever
                        metric = line['metric'].replace('_' + pt_retriever, '')
                correlation_per_query = line['correlation_per_query']

                key1 = get_key([eva_met, retriever])
                key2 = get_key([agr_met, tra_met, metric])

                # Correlation scores for each evaluation method and retriever
                if key1 not in correlation_scores_eva_ret:
                    correlation_scores_eva_ret[key1] = {}
                if key2 in correlation_scores_eva_ret[key1]:
                    print('Dublicate:', key1, key2)
                correlation_scores_eva_ret[key1][key2] = correlation_per_query

    print(f"Files: {files}")


def count_scores_per_qid():
    path = '/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-fhofer/data/argsme/2020-04-01-backup/touche\
-2021-task-1/backup/retrieval-scores-aq.jsonl.gz'

    with gzip.open(path, 'rt', encoding='UTF-8') as file:
        total = 0
        scores = {}
        for line in tqdm(file, desc='Caching scores', unit='score'):
            line = json.loads(line)
            qid = line['qid']
            if qid not in scores:
                scores[qid] = 0
            scores[qid] += 1
            total += 1

    print(scores)
    print(f"Total: {total}")
    print(f"Qids: {len(scores)}")


def test_clueweb_access():
    dataset = ir_datasets.load("clueweb12/trec-web-2013")
    docstore = dataset.docs_store()
    print(docstore.get('clueweb12-0006wb-88-26772').default_text())


def check_labels():
    DOCUMENT_DATASET_SOURCE_NAME = config['DOCUMENT_DATASET_SOURCE_NAME']
    SOURCE_PATH = os.path.join(config['DATA_PATH'], DOCUMENT_DATASET_SOURCE_NAME)
    PASSAGE_DATASET_SOURCE_SCORE_REL_PATH = os.path.join(SOURCE_PATH, config['PASSAGE_DATASET_SOURCE_SCORE_REL_PATH'])
    PASSAGE_DATASET_SOURCE_SCORE_AQ_PATH = os.path.join(SOURCE_PATH, config['PASSAGE_DATASET_SOURCE_SCORE_AQ_PATH'])
    FILE_PATTERN_REL = os.path.join(PASSAGE_DATASET_SOURCE_SCORE_REL_PATH, "qid_*.jsonl.gz")
    FILE_PATTERN_AQ = os.path.join(PASSAGE_DATASET_SOURCE_SCORE_AQ_PATH, "qid_*.jsonl.gz")

    for file_path in glob(FILE_PATTERN_REL):
        with gzip.open(file_path, 'rt', encoding='UTF-8') as file:
            for line in file:
                line = json.loads(line)
                if line['label'] < 0:
                    print(line)

    for file_path in glob(FILE_PATTERN_AQ):
        with gzip.open(file_path, 'rt', encoding='UTF-8') as file:
            for line in file:
                line = json.loads(line)
                if line['label'] < 0:
                    print(line)


test_clueweb_access()
