import gzip
import json
import pyterrier as pt
import os


# Load the configuration settings
def load_config(filename="../config.json"):
    with open(filename, "r") as f:
        config = json.load(f)
    return config


# Get the configuration settings
config = load_config()

DOCUMENT_DATASET_OLD_NAME = config['DOCUMENT_DATASET_OLD_NAME']
DOCUMENT_DATASET_OLD_NAME_PYTERRIER = config['DOCUMENT_DATASET_OLD_NAME_PYTERRIER']

OLD_PATH = os.path.join(config['DATA_PATH'], DOCUMENT_DATASET_OLD_NAME)

NUMBER_OF_CROSS_VALIDATION_FOLDS = config['NUMBER_OF_CROSS_VALIDATION_FOLDS']
KEY_SEPARATOR = config['KEY_SEPARATOR']

PASSAGES_TO_DOCUMENT_CORRELATION_SCORE_PATH = os.path.join(
    OLD_PATH, config['PASSAGES_TO_DOCUMENT_CORRELATION_SCORE_PQ_AQ_PATH'])


# 1. Load correlation scores
#    KEY: agr_method___trans_method___eval_method___metric_retriever
#    SAVING: {key: {qid: score}}
#    ACCESS: correlation_scores[key] = {qid: score}
def get_key(list):
    return KEY_SEPARATOR.join(list)


correlation_scores = {}
with gzip.open(PASSAGES_TO_DOCUMENT_CORRELATION_SCORE_PATH, 'rt', encoding='UTF-8') as file:
    for line in file:
        line = json.loads(line)
        agr_met = line['aggregation_method']
        tra_met = line['transformation_method']
        eva_met = line['evaluation_method']
        met_ret = line['metric']  # eg p10_BM25
        correlation_per_query = line['correlation_per_query']

        key = get_key([agr_met, tra_met, eva_met, met_ret])

        if key not in correlation_scores:
            correlation_scores[key] = correlation_per_query


# 2. Load qrels and sort by qid
#    SAVING: {qid: {docno: label}}
#    ACCESS: qrels[qid] = pd.DataFrame with columns [qid docno label quality iteration]
# dataset = pt.get_dataset(DOCUMENT_DATASET_OLD_NAME_PYTERRIER)
# qrels = dataset.get_qrels(variant='relevance')
# qrels = qrels.sort_values(by=['qid'])  # sort by qid
# qrels_cache = {}
# for index, row in qrels.iterrows():
#     if row['qid'] not in qrels_cache:
#         qrels_cache[row['qid']] = qrels.loc[
#             (qrels['qid'] == row['qid'])
#         ]

# 2. get all qids
qids = []
dataset = pt.get_dataset(DOCUMENT_DATASET_OLD_NAME_PYTERRIER)
for query in dataset.irds_ref().queries_iter():
    qids.append(query.query_id)

# 3. N-fold cross validation
k = NUMBER_OF_CROSS_VALIDATION_FOLDS
fold_size = len(qids) // k
for i in range(k):
    test_qids = qids[i * fold_size:(i + 1) * fold_size]
    train_qids = qids[:i * fold_size] + qids[(i + 1) * fold_size:]

    print(test_qids)
    print(train_qids)
    print('---------------------------------------')
