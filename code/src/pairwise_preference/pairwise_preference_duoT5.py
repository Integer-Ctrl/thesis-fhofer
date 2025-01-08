import autoqrels
import gzip
import json
import os
import ir_datasets
import time
from tqdm import tqdm
import torch

# Load the configuration settings
pwd = os.path.dirname(os.path.abspath(__file__))


def load_config(filename=pwd + "/../config.json"):
    with open(filename, "r") as f:
        config = json.load(f)
    return config


# Get the configuration settings
config = load_config()

DOCUMENT_DATASET_TARGET_NAME = config['DOCUMENT_DATASET_TARGET_NAME']
DOCUMENT_DATASET_TARGET_NAME_PYTHON_API = config['DOCUMENT_DATASET_TARGET_NAME_PYTHON_API']

SOURCE_PATH = os.path.join(config['DATA_PATH'], config["DOCUMENT_DATASET_SOURCE_NAME"])
TARGET_PATH = os.path.join(SOURCE_PATH, config["DOCUMENT_DATASET_TARGET_NAME"])

CANDIDATES_PATH = os.path.join(TARGET_PATH, config['CANDIDATES_LOCAL_PATH'], 'nearest_neighbor.jsonl.gz')
DUOT5_CACHE_PATH = os.path.join(TARGET_PATH, config['DUOT5_CACHE_PATH'])

KEY_SEPARATOR = config['KEY_SEPARATOR']


# Helper function to access the cache
def get_key(list):
    return KEY_SEPARATOR.join(list)


# Check if cache exists and load it if it does
# Key: qid---known_relevant_passage_id---upassage_to_judge_id: score
candidates_cache = {}
if os.path.exists(DUOT5_CACHE_PATH):
    with gzip.open(DUOT5_CACHE_PATH, 'rt') as file:
        for line in file:
            line = json.loads(line)
            key = get_key([line['qid'], line['known_relevant_passage_id'], line['passage_to_judge_id']])
            candidates_cache[key] = line['score']

grouped_candidates = {}
with gzip.open(CANDIDATES_PATH, 'rt') as file:
    for line in file:
        # Add all candidates to the list, also those that are already in the cache due to the cache is overwritten
        candidate = json.loads(line)
        key = get_key([candidate['qid'],
                       candidate['known_relevant_passage']['docno']])
        if key not in grouped_candidates:
            grouped_candidates[key] = []
        grouped_candidates[key].append(candidate)

# Load the dataset and the DuoPrompt model
dataset = ir_datasets.load(DOCUMENT_DATASET_TARGET_NAME_PYTHON_API)
duoprompt = autoqrels.oneshot.DuoT5(dataset=dataset, device='cuda', batch_size=8)

# Iterate over the grouped candidates and infer the relevance
scores = []
for key, group in tqdm(grouped_candidates.items(), desc="Infer relevance"):

    qid, known_relevant_passage = key.split(KEY_SEPARATOR)
    query_text = group[0]['query']
    rel_doc_id = group[0]['known_relevant_passage']['docno']
    rel_doc_text = group[0]['known_relevant_passage']['text']

    unk_doc_scores = {}
    unk_doc_ids = []
    unk_doc_texts = []

    start_time = time.time()
    pairwise_pref_count = 0

    for candidate in group:

        # TODO: Remove safety check
        if candidate['qid'] != qid:
            print("Error: qid mismatch")
            exit()

        # If the score is already in the cache, dont infer it again
        key = get_key([candidate['qid'],
                       candidate['known_relevant_passage']['docno'],
                       candidate['passage_to_judge']['docno']])
        if key in candidates_cache:
            continue

        # If the score is not in the cache, add the unknown document to the list
        unk_doc_ids.append(candidate['passage_to_judge']['docno'])
        unk_doc_texts.append(candidate['passage_to_judge']['text'])
        pairwise_pref_count += 1

    # If there are no unknown documents to infer, continue to the next group
    if len(unk_doc_ids) == 0:
        continue

    # Infer the relevance of the unknown documents
    inferred_scores = duoprompt.infer_oneshot_text(query_text=query_text,
                                                   rel_doc_text=rel_doc_text,
                                                   unk_doc_texts=unk_doc_texts)
    time_elapsed = time.time() - start_time
    print(f"Infered {pairwise_pref_count} pairwise preferences in {time_elapsed / 60} minutes")
    print(f"    qid: {qid}, known_relevant_passage: {known_relevant_passage}")

    # TODO: Remove safety check
    if len(unk_doc_ids) != len(inferred_scores):
        print("Error: Inferred scores mismatch")
        exit()

    for unk_doc_id, score in zip(unk_doc_ids, inferred_scores):
        unk_doc_scores[unk_doc_id] = score

    # Write the scores to the cache
    with gzip.open(DUOT5_CACHE_PATH, 'at') as file:
        for unk_doc_id, score in unk_doc_scores.items():
            file.write(json.dumps({"qid": qid,
                                   "known_relevant_passage_id": known_relevant_passage,
                                   "passage_to_judge_id": unk_doc_id,
                                   "score": score}) + '\n')
