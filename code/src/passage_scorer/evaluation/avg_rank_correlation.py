import gzip
import json
import pyterrier as pt
import os
from glob import glob


# Load the configuration settings
pwd = os.path.dirname(os.path.abspath(__file__))


def load_config(filename=pwd + "/../../config.json"):
    with open(filename, "r") as f:
        config = json.load(f)
    return config


# Get the configuration settings
config = load_config()

DOCUMENT_DATASET_SOURCE_NAME = config['DOCUMENT_DATASET_SOURCE_NAME']
SOURCE_PATH = os.path.join(config['DATA_PATH'], DOCUMENT_DATASET_SOURCE_NAME)
RANK_CORRELATION_SCORE_AVG_PATH = os.path.join(SOURCE_PATH, config['RANK_CORRELATION_SCORE_AVG_PATH'])
RANK_CORRELATION_SCORE_PQ_AQ_PATTERN = os.path.join(SOURCE_PATH, config['RANK_CORRELATION_SCORE_PQ_AQ_PATH'], '*.jsonl.gz')
METRICS = ['ndcg10', 'p10']  # only these metrics are considered for further processing

avg_correlation_score = {}

for file_name in glob(RANK_CORRELATION_SCORE_PQ_AQ_PATTERN):
    with gzip.open(file_name, 'rt', encoding='UTF-8') as file:
        for line in file:
            line = json.loads(line)

            aggregation_method = line['aggregation_method']
            transformation_method = line['transformation_method']
            evaluation_method = line['evaluation_method']
            metric_retriever = line['metric']

            for metric in METRICS:
                # Filter out the metrics that are not in the METRICS list
                if metric in metric_retriever:
                    if 'wod' in metric_retriever:
                        continue
                    metric = metric
                    retriever = metric_retriever.replace(f'{metric}_', '')

                    correlation_per_query = line['correlation_per_query']

                    key = f"{evaluation_method}---{aggregation_method}---{transformation_method}---{retriever}---{metric}"
                    if key in avg_correlation_score:
                        print(f"ERROR: dublicate key {key}")
                    
                    avg_score = sum(correlation_per_query.values()) / len(correlation_per_query)
                    avg_correlation_score[key] = avg_score

# Sort the dictionary by value
sorted_avg_correlation_score = dict(sorted(avg_correlation_score.items(), key=lambda item: item[1], reverse=True))

with gzip.open(RANK_CORRELATION_SCORE_AVG_PATH, 'wt', encoding='UTF-8') as file:
    for key, value in sorted_avg_correlation_score.items():
        file.write(json.dumps({"eval---aggr---tra---retriever---metric": key, "score": value}) + "\n")