import autoqrels
import autoqrels.zeroshot
import pandas as pd
from typing import List

# Sample data for 'run' DataFrame
data_run = {
    'query_id': ['q1', 'q1', 'q1', 'q1', 'q2', 'q2', 'q2', 'q2', 'q3', 'q3', 'q3', 'q3', 'q4', 'q4', 'q4', 'q4'],
    'doc_id': ['d1', 'd2', 'd3', 'd4', 'd1', 'd2', 'd3', 'd4', 'd1', 'd2', 'd3', 'd4', 'd1', 'd2', 'd3', 'd4'],
    'score': [0.9, 0.8, 0.7, 0.85, 0.2, 0.3, 0.7, 0.8, 0.9, 0.5, 0.5, 0.6, 0.2, 0.8, 0.4, 0.9, ]
}
df_run = pd.DataFrame(data_run)

# Sample data for 'qrels' DataFrame
data_qrels = {
    'query_id': ['q1', 'q2', 'q3', 'q4'],
    'doc_id': ['d1', 'd3', 'd4', 'd1'],
    'relevance': [1, 1, 1, 1]  # Only one relevant doc per query for OneShotLabeler
}
df_qrels = pd.DataFrame(data_qrels)


# zero shot labeler example
def zero_shot_labeler_example(data_run):

    def mock_infer_qrels(query_id: str, unk_doc_ids: List[str]) -> List[float]:
        # Assign alternating relevance scores of 0 and 1 to the unknown documents
        return [i % 2 for i in range(len(unk_doc_ids))]

    labeler = autoqrels.zeroshot.ZeroShotLabeler()
    labeler._infer_zeroshot = mock_infer_qrels
    result = labeler.infer_qrels(data_run)
    return result


# one shot labeler example
def one_shot_labeler_example(data_run, data_qrels):
    # cache structure for zero-shot inferences: {query_id: {unk_doc_id: score}}

    def mock_infer_qrels(query_id: str, rel_doc_id: str, unk_doc_ids: List[str]) -> List[float]:
        # Get the 'run' data for the specified query
        run_group = data_run[data_run['query_id'] == query_id]

        # Filter out the relevant document (rel_doc_id)
        unk_docs_scores = run_group[run_group['doc_id'].isin(unk_doc_ids)]

        # Get score for the relevant document
        rel_score = run_group[run_group['doc_id'] == rel_doc_id]['score'].values[0]

        # Get the scores for the unknown documents
        scores = unk_docs_scores['score'].values

        return [1.0 if score > rel_score else 0.0 for score in scores]

    labeler = autoqrels.oneshot.OneShotLabeler()
    labeler._infer_oneshot = mock_infer_qrels
    result = labeler.infer_qrels(data_run, data_qrels)
    return result


# examples
result = zero_shot_labeler_example(df_run)
print("Zero shot example:\n", result)

result = one_shot_labeler_example(df_run, df_qrels)
print("One shot example:\n", result)
