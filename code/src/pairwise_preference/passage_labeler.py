# 1. get all passages in dictionary format docno: text
# 2. get type of best scoring metric in rank correlation TODO: implement this more dynamic (save file with prefixes)
# 3. get all passage scores in dictionary format qid: {docno: score} # just score of the best scoring method
# 4. get all qrels in dictinary format qid: {docno: relevance} # all relevance scores
# 5. get all queries in dictionary format query_id: text
# 6. get for all queries the best passages in dictionary format query_id: [docno] without duplicates docno
# 7. get all passages to judge in dictionary format qid: [docno]
import json
import pyterrier as pt
import os
import gzip
from tqdm import tqdm
import ir_datasets
from transformers import T5ForConditionalGeneration, T5Tokenizer
from duoT5_inference import RelevanceInference


# Load the configuration settings
# def load_config(filename="../config.json"): does not work with debug
def load_config(filename="/mnt/ceph/storage/data-tmp/current/ho62zoq/thesis-fhofer/code/src/config.json"):
    with open(filename, "r") as f:
        config = json.load(f)
    return config


# Get the configuration settings
config = load_config()

DOCUMENT_DATASET_OLD_NAME = config['DOCUMENT_DATASET_OLD_NAME']
DOCUMENT_DATASET_OLD_NAME_PYTERRIER = config['DOCUMENT_DATASET_OLD_NAME_PYTERRIER']
DOCUMENT_DATASET_OLD_NAME_PYTHON_API = config['DOCUMENT_DATASET_OLD_NAME_PYTHON_API']

DATA_PATH = os.path.join(config['DATA_PATH'], DOCUMENT_DATASET_OLD_NAME)

PASSAGE_DATASET_PATH = os.path.join(DATA_PATH, config['PASSAGE_DATASET_PATH'])
PASSAGE_DATASET_SCORE_PATH = os.path.join(DATA_PATH, config['PASSAGE_DATASET_SCORE_PATH'])
# TODO: switch between AQ and PQ
PASSAGES_TO_DOCUMENT_CORRELATION_SCORE_PATH = os.path.join(
    DATA_PATH, config['PASSAGES_TO_DOCUMENT_CORRELATION_SCORE_PATH'])
PASSAGE_ID_SEPARATOR = config['PASSAGE_ID_SEPARATOR']

KEY_SEPARATOR = config['KEY_SEPARATOR']
DUOT5_QID_DOC_DOC_SYSTEM_SCORES_PATH = os.path.join(DATA_PATH, config['DUOT5_QID_DOC_DOC_SYSTEM_SCORES_PATH'])


# Read passages and cache them
passages_text_cache = {}
passages_score_cache = {}
qrels_cache = {}
queries_cache = {}
queries_best_passages_cache = {}
documents_to_judge_cache = {}
pairwise_cache = {}


def get_key(query_id: str, rel_doc_id: str, unk_doc_id: str, system: str) -> str:
    return f"{query_id}___{rel_doc_id}___{unk_doc_id}___{system}"


# 1. get all passages in dictionary format docno: {passageno: text}
def get_passages_text(cache):
    with gzip.open(PASSAGE_DATASET_PATH, 'rt', encoding='UTF-8') as file:
        for line in tqdm(file, desc='Caching passages', unit='passage'):
            line = json.loads(line)
            docno, _ = line['docno'].split(PASSAGE_ID_SEPARATOR)
            if docno not in cache:
                cache[docno] = []
            cache[docno] += [line]


# 2. get type of best scoring metric in rank correlation
def get_best_scoring_methods():
    with gzip.open(PASSAGES_TO_DOCUMENT_CORRELATION_SCORE_PATH, 'rt', encoding='UTF-8') as file:
        for line in file:  # already decending sorted
            return json.loads(line)  # return only best scoring method


# 3. get all passage scores in dictionary format qid: {docno: score} # just score of the best scoring method
def get_passages_scores(cache, metric):
    with gzip.open(PASSAGE_DATASET_SCORE_PATH, 'rt', encoding='UTF-8') as file:
        for line in tqdm(file, desc='Caching passage scores', unit='passage'):
            data = json.loads(line)
            qid = data['qid']        # Extract query ID
            docno = data['docno']    # Extract document number

            # Store the best score in the cache
            if qid not in cache:
                cache[qid] = {}
            cache[qid][docno] = data[metric]


# 4. get all qrels in dictinary format qid: {docno: relevance} # all relevance scores
def get_qrels(cache):
    dataset = pt.get_dataset(DOCUMENT_DATASET_OLD_NAME_PYTERRIER)
    qrels = dataset.get_qrels()
    for index, row in tqdm(qrels.iterrows(), desc='Caching qrels', unit='qrel'):
        if row['qid'] not in cache:
            cache[row['qid']] = {}
        cache[row['qid']][row['docno']] = row['label']


# 5. get all queries in dictionary format query_id: text
def get_queries(queries_cache):
    dataset = ir_datasets.load(DOCUMENT_DATASET_OLD_NAME_PYTHON_API)
    for query in dataset.queries_iter():
        queries_cache[query.query_id] = query.default_text()


# 6. get for all queries the best passages in dictionary format query_id: [docno] without duplicates docno
def get_queries_best_passages_one_per_document(cache, scores):
    for qid, passageno_scores in scores.items():
        # Step 1: Parse docnos and sort by score
        docnos_best_passagenos = {}
        for passageno, score in passageno_scores.items():
            # Extract docno by removing the suffix ___x
            docno, _ = passageno.split(PASSAGE_ID_SEPARATOR)
            # Keep the highest-scoring passageno for each docno
            if docno not in docnos_best_passagenos or score > docnos_best_passagenos[docno][1]:
                docnos_best_passagenos[docno] = (passageno, score)

        # Step 2: Extract highest-scored passagenos and sort them in descending order
        best_passagenos = [item[0]
                           for item in sorted(docnos_best_passagenos.values(), key=lambda x: x[1], reverse=True)]

        # Add to result
        cache[qid] = best_passagenos


# 7. get all pairwise scores in dictionary format qid: {qid, docno1, docno2, system, score}
def read_pairwise_cache():
    cache = {}
    if os.path.isfile(DUOT5_QID_DOC_DOC_SYSTEM_SCORES_PATH):
        with gzip.open(DUOT5_QID_DOC_DOC_SYSTEM_SCORES_PATH, 'rt', encoding='UTF-8') as file:
            for line in file:
                line = json.loads(line)
                key = get_key(line['qid'], line['rel_doc_id'], line['unk_doc_id'], line['system'])
                cache[key] = line['score']
    return cache


# save cache to file
def save_pairwise_cache(cache):
    with gzip.open(DUOT5_QID_DOC_DOC_SYSTEM_SCORES_PATH, 'wt', encoding='UTF-8') as file:
        for key, score in cache.items():
            query_id, rel_doc_id, unk_doc_id, system = key.split(KEY_SEPARATOR)
            file.write(json.dumps({
                'qid': query_id,
                'rel_doc_id': rel_doc_id,
                'unk_doc_id': unk_doc_id,
                'system': system,
                'score': score
            }) + '\n')


# Init all caches and read data
get_passages_text(passages_text_cache)
best_scoring_method = get_best_scoring_methods()
get_passages_scores(passages_score_cache, best_scoring_method['metric'])
get_qrels(qrels_cache)
get_queries(queries_cache)
read_pairwise_cache()

# Get for all queries the best passages in dictionary format query_id: [docno] without passages with same docno
get_queries_best_passages_one_per_document(queries_best_passages_cache, passages_score_cache)

# Init inference
model_name = 'castorini/duot5-base-msmarco'
tokeniser_name = 't5-base'
model = T5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(tokeniser_name)

inference = RelevanceInference(model, model_name, tokenizer, queries_cache, passages_text_cache, pairwise_cache)


# 1. APPROACH: this approach will likely overestimate
count = 0
for qid, docno_scores in documents_to_judge_cache.items():
    for docno, score in docno_scores.items():
        count += 1
        # query_id = qid
        # unk_doc_id = docno
        # rel_doc_ids = queries_best_passages_cache[qid]
        # rel_doc_ids = rel_doc_ids[:20]

        # # infer relevance scores
        # scores = inference._infer_oneshot(query_id, unk_doc_id, rel_doc_ids)
        # score = sum(scores) / len(scores)
        # print(score)
        # # save to cache
        # save_pairwise_cache(pairwise_cache)
        # exit()
print(count)


# 2. APPROACH TODO: for each query get top 2000 bm25 passages and infer relevance scores
# 3. APPROACH TODO: for each query get top 2000 bm25 passages + rel_doc_ids[:20] first 10 as query
#  and top 10 bm25 retrieved passagen and infer relevance scores

# TODO: evaluate retrieved passages with Recall and Precision (Recall more important)
# as indicator, all passages with relevance label > 0 are good
