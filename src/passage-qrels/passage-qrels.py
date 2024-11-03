import autoqrels
import autoqrels.zeroshot
import pandas as pd
from typing import List
import gzip
import json
import pyterrier as pt

PASSAGE_SCORES_PATH = '../data/argsme/passage-dataset/bm25-scores.jsonl.gz'
PASSAGE_QRELS_PATH = '../data/argsme/passage-dataset/qrels.jsonl.gz'
DATASET_ID = 'irds:argsme/2020-04-01/touche-2020-task-1'


def get_passage_scores():
    with gzip.open(PASSAGE_SCORES_PATH, 'rt', encoding='UTF-8') as file:
        scores = pd.read_json(file, lines=True)
        return scores.rename(columns={'qid': 'query_id', 'docno': 'doc_id', 'p10': 'score'})


def get_document_qrels():
    dataset = pt.get_dataset(DATASET_ID)
    qrels = dataset.get_qrels()
    # filter out only relevant documents
    qrels = qrels[qrels['label'] > 0]
    return qrels.rename(columns={'qid': 'query_id', 'docno': 'doc_id', 'label': 'relevance'})


def zero_shot_labeler(run):

    def mock_infer_qrels(query_id: str, unk_doc_ids: List[str]) -> List[float]:
        # Filter the DataFrame for the specified query_id and doc_ids
        query_run = run[(run['query_id'] == query_id) & (run['doc_id'].isin(unk_doc_ids))]

        # Classify each document as relevant (1) if score > 0.8, else not relevant (0)
        qrels = [1 if score >= 0.6 else 0 for score in query_run['score']]

        return qrels

    labeler = autoqrels.zeroshot.ZeroShotLabeler()
    labeler._infer_zeroshot = mock_infer_qrels
    result = labeler.infer_qrels(run)
    return result


def one_shot_labeler(run, qrels):
    # cache structure for zero-shot inferences: {query_id: {unk_doc_id: score}}

    def mock_infer_qrels(query_id: str, rel_doc_id: str, unk_doc_ids: List[str]) -> List[float]:
        # TODO: "oneshot labelers only support one relevant qrel per query"
        # use zero-shot inference for now
        # idea to use one-shot inference:
        # compare the score of the relevant document with the scores of the unknown documents
        pass

    labeler = autoqrels.oneshot.OneShotLabeler()
    labeler._infer_oneshot = mock_infer_qrels
    result = labeler.infer_qrels(run, qrels)
    return result


# qrels = get_document_qrels() # TODO: implement one-shot inference
run = get_passage_scores()
# result = one_shot_labeler(run, qrels)
result = zero_shot_labeler(run)

with gzip.open(PASSAGE_QRELS_PATH, 'at', encoding='UTF-8') as f_out:
    for _, row in result.iterrows():
        # Check if relevance is greater than or equal to 0.6
        if row['relevance'] >= 1:
            # Convert the row to a dictionary
            row_dict = row.to_dict()
            # Write the dictionary as a JSON line
            f_out.write(json.dumps(row_dict) + '\n')
