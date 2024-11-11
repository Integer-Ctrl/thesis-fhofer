import autoqrels
import autoqrels.zeroshot
import pandas as pd
from typing import List
import gzip
import json
import numpy as np
from tqdm import tqdm
import copy
import sys

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
print(qrels)
