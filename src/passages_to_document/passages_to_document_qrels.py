import autoqrels
import autoqrels.zeroshot
import pandas as pd
from typing import List
import gzip
import json
import numpy as np
from tqdm import tqdm
import copy
import pyterrier as pt
from sklearn.metrics import cohen_kappa_score

DATASET_NAME = 'irds:argsme/2020-04-01/touche-2021-task-1'
PASSAGE_SCORES_PATH = '../data/' + DATASET_NAME.replace('irds:', '') + '/passage-dataset/passage-scores.jsonl.gz'
PASSAGE_TO_DOCUMENT_CORRELATION_SCORES_PATH = '../data/' + \
    DATASET_NAME.replace('irds:', '') + '/document-dataset/passages-to-document/correlation-scores.jsonl.gz'


# Read passsage scores and cache them
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


def get_best_scoring_methods():
    with gzip.open(PASSAGE_TO_DOCUMENT_CORRELATION_SCORES_PATH, 'rt', encoding='UTF-8') as file:
        for line in file:  # already decending sorted
            return json.loads(line)  # return only best scoring method


dataset = pt.get_dataset(DATASET_NAME)
qrels = dataset.get_qrels()
qrels_cache = {}
for index, row in tqdm(qrels.iterrows(), desc='Caching qrels', unit='qrel'):
    # Only relevant qrels
    if row['label'] > 0:
        if row['qid'] not in qrels_cache:
            qrels_cache[row['qid']] = qrels.loc[
                (qrels['qid'] == row['qid']) & (qrels['label'] > 0)  # All relevant entries for the query ID
            ]


# Function to get dictonary of aggregated score for a document using passage scores
# Return a list of dictionaries with aggregated scores for each document {docno, qid, metric: score}
def get_docno_qid_aggregated_scores(docno_qid_passages_scores_cache, aggregation_method, metric):
    # All metrics that are available in the passage scores

    aggregated_scores = []

    for docno, qid_passages_scores in docno_qid_passages_scores_cache.items():
        for qid, passages_scores in qid_passages_scores.items():

            aggregated_doc_scores = {'docno': docno, 'qid': qid}
            scores = [passage[metric] for passage in passages_scores]

            if aggregation_method == 'mean':
                aggregated_doc_scores[metric] = float(np.mean(scores))
            elif aggregation_method == 'max':
                aggregated_doc_scores[metric] = float(np.max(scores))
            elif aggregation_method == 'min':
                aggregated_doc_scores[metric] = float(np.min(scores))

            aggregated_scores.append(aggregated_doc_scores)

    return aggregated_scores


# Function to get transformed scores
def get_docno_qid_transformed_scores(docno_qid_aggregated_scores, transformation_method, metric, bins=[0.3, 0.7]):

    MIN_CORRELATION_SCORE = float('inf')
    MAX_CORRELATION_SCORE = float('-inf')

    docno_qid_aggregated_scores_transformed = copy.deepcopy(docno_qid_aggregated_scores)
    for entry in docno_qid_aggregated_scores_transformed:

        if transformation_method == 'id':
            pass
        elif transformation_method == 'log' and entry[metric] != 0:
            entry[metric] = float(np.log(entry[metric]))
        elif transformation_method == 'binned':
            entry[metric] = float(np.digitize(entry[metric], bins))

        if entry[metric] < MIN_CORRELATION_SCORE:
            MIN_CORRELATION_SCORE = entry[metric]
        if entry[metric] > MAX_CORRELATION_SCORE:
            MAX_CORRELATION_SCORE = entry[metric]

    return docno_qid_aggregated_scores_transformed, MIN_CORRELATION_SCORE, MAX_CORRELATION_SCORE


def zero_shot_labeler(run, boundaries):

    def mock_infer_qrels(query_id: str, unk_doc_ids: List[str]) -> List[float]:
        # Filter the DataFrame for the specified query_id and doc_ids
        query_run = run[(run['query_id'] == query_id) & (run['doc_id'].isin(unk_doc_ids))]

        # Classify each document as relevant (1) if score > 0.8, else not relevant (0)
        qrels = [
            0 if score < boundaries[0] else
            1 if boundaries[0] <= score <= boundaries[1] else
            2
            for score in query_run['score']
        ]

        return qrels

    labeler = autoqrels.zeroshot.ZeroShotLabeler()
    labeler._infer_zeroshot = mock_infer_qrels
    result = labeler.infer_qrels(run)
    return result


# Calculate Cohen's Kappa
def calculate_cohen_kappa(qrels, qrels_cache):
    # Lists to store relevance values and labels
    relevance_array = []
    label_array = []

    # Iterate over the qrels dataframe row by row
    for _, row in qrels.iterrows():
        query_id = row['query_id']
        doc_id = row['doc_id']
        relevance = row['relevance']

        # Include all relevance labels, also 0
        # Check if the query_id exists in qrels_cache
        if query_id in qrels_cache:
            # Get the cached qrels for this query_id
            cached_qrels = qrels_cache[query_id]

            # Find the row in cached_qrels that matches the current doc_id
            cached_row = cached_qrels[cached_qrels['docno'] == doc_id]

            if not cached_row.empty:
                # Get the label (assuming single match per doc_id)
                label = cached_row['label'].values[0]

                # Append to arrays
                relevance_array.append(relevance)
                label_array.append(label)

    # Convert lists to numpy arrays
    relevance_array = np.array(relevance_array)
    label_array = np.array(label_array)

    return cohen_kappa_score(relevance_array, label_array)


best_scoring_methods = get_best_scoring_methods()
docno_qid_aggregated_scores = get_docno_qid_aggregated_scores(
    docno_qid_passages_scores_cache, best_scoring_methods['aggregation_method'], best_scoring_methods['metric'])
docno_qid_transformed_scores, min, max = get_docno_qid_transformed_scores(
    docno_qid_aggregated_scores, best_scoring_methods['transformation_method'], best_scoring_methods['metric'])

range = max - min
boundaries = [min + 0.4 * range, min + 0.7 * range]  # relevance 0, 1 and 2


run = pd.DataFrame(docno_qid_transformed_scores).rename(
    columns={'docno': 'doc_id', 'qid': 'query_id', best_scoring_methods['metric']: 'score'})

qrels = zero_shot_labeler(run, boundaries)

score = calculate_cohen_kappa(qrels, qrels_cache)
print(score)
