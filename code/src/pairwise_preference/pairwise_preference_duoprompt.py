import autoqrels
import gzip
import json
import os
import ir_datasets
import time
from tqdm import tqdm
from glob import glob

# Load the configuration settings
pwd = os.path.dirname(os.path.abspath(__file__))


def load_config(filename=pwd + "/../config.json"):
    with open(filename, "r") as f:
        config = json.load(f)
    return config


# Get the configuration settings
config = load_config()

CHATNOIR_RETRIEVAL = config['CHATNOIR_RETRIEVAL']

DOCUMENT_DATASET_TARGET_NAME = config['DOCUMENT_DATASET_TARGET_NAME']
DOCUMENT_DATASET_TARGET_NAME_PYTHON_API = config['DOCUMENT_DATASET_TARGET_NAME_PYTHON_API']

SOURCE_PATH = os.path.join(config['DATA_PATH'], config["DOCUMENT_DATASET_SOURCE_NAME"])
TARGET_PATH = os.path.join(SOURCE_PATH, config["DOCUMENT_DATASET_TARGET_NAME"])

if CHATNOIR_RETRIEVAL:
    CANDIDATES_PATH = os.path.join(TARGET_PATH, config['CANDIDATE_CHATNOIR_PATH'])
else:
    CANDIDATES_PATH = os.path.join(TARGET_PATH, config['CANDIDATES_LOCAL_PATH'])
FILE_PATTERN = os.path.join(CANDIDATES_PATH, "*.jsonl.gz")

DUOPROMPT_PATH = os.path.join(TARGET_PATH, config['DUOPROMPT_PATH'])
DUOPROMPT_CACHE = os.path.join(DUOPROMPT_PATH, config['DUOPROMPT_CACHE_NAME'])

KEY_SEPARATOR = config['KEY_SEPARATOR']


# Helper function to access the cache
def get_key(list):
    return KEY_SEPARATOR.join(list)


# Check if cache exists and load it if it does
# Key: qid---known_passage_id---upassage_to_judge_id: score
# Known passage is either known relevant or known non relevant
candidates_cache = {}
if os.path.exists(DUOPROMPT_CACHE):
    with gzip.open(DUOPROMPT_CACHE, 'rt') as file:
        for line in file:
            line = json.loads(line)

            key = get_key([line['qid'], line['known_passage_id'], line['passage_to_judge_id']])
            candidates_cache[key] = line['score']


def process_candidates(candidates_path, pairwise_preferences_path):

    # Load the dataset and the DuoPrompt model
    dataset = ir_datasets.load(DOCUMENT_DATASET_TARGET_NAME_PYTHON_API)
    duoprompt = autoqrels.oneshot.DuoPrompt(dataset=dataset,
                                            backbone='google/flan-t5-base',
                                            device='cuda',
                                            batch_size=32)

    used_cached_count = 0
    infered_count = 0

    grouped_candidates = {}
    grouped_relevant_candidates = {}
    grouped_non_relevant_candidates = {}

    with gzip.open(candidates_path, 'rt') as file:
        for line in file:
            # Add all candidates to the list, also those that are already in the cache due to the cache is overwritten
            candidate = json.loads(line)

            if candidate['known_relevant_passage']:
                key = get_key([candidate['qid'],
                               candidate['known_relevant_passage']['docno']])
                # Add candidate to all grouped candidates
                if key not in grouped_candidates:
                    grouped_candidates[key] = []
                grouped_candidates[key].append(candidate)
                # Add candidate to grouped relevant candidates
                if key not in grouped_relevant_candidates:
                    grouped_relevant_candidates[key] = []
                grouped_relevant_candidates[key].append(candidate)

            elif candidate['known_non_relevant_passage']:
                key = get_key([candidate['qid'],
                               candidate['known_non_relevant_passage']['docno']])
                # Add candidate to all grouped candidates
                if key not in grouped_candidates:
                    grouped_candidates[key] = []
                grouped_candidates[key].append(candidate)
                # Add candidate to grouped non relevant candidates
                if key not in grouped_non_relevant_candidates:
                    grouped_non_relevant_candidates[key] = []
                grouped_non_relevant_candidates[key].append(candidate)

            else:
                print("Error: Candidate is neither known relevant nor known non relevant")
                exit()

    # Check if candidates are in cache already and if not infer them
    # Iterate over the grouped candidates and infer the relevance
    for key, group in tqdm(grouped_candidates.items(), desc="Infer relevance"):

        qid, known_passage_id = key.split(KEY_SEPARATOR)
        query_text = group[0]['query_text']

        unk_doc_scores = {}
        unk_doc_ids = []
        unk_doc_texts = []

        for candidate in group:

            # TODO: Remove safety check
            if candidate['qid'] != qid:
                print("Error: qid mismatch")
                exit()

            # If the score is already in the cache, dont infer it again
            if candidate['known_relevant_passage']:
                known_doc_text = candidate['known_relevant_passage']['text']
                key = get_key([candidate['qid'],
                               candidate['known_relevant_passage']['docno'],
                               candidate['passage_to_judge']['docno']])
            elif candidate['known_non_relevant_passage']:
                known_doc_text = candidate['known_non_relevant_passage']['text']
                key = get_key([candidate['qid'],
                               candidate['known_non_relevant_passage']['docno'],
                               candidate['passage_to_judge']['docno']])
            else:
                print("Error: Candidate is neither known relevant nor known non relevant")
                exit()

            if key in candidates_cache:
                used_cached_count += 1
                continue

            # If the score is not in the cache, add the unknown document to the list
            unk_doc_ids.append(candidate['passage_to_judge']['docno'])
            unk_doc_texts.append(candidate['passage_to_judge']['text'])
            infered_count += 1

        # If there are no unknown documents to infer, continue to the next group
        if len(unk_doc_ids) == 0:
            continue

        # Infer the relevance of the unknown documents
        inferred_scores = duoprompt.infer_oneshot_text(query_text=query_text,
                                                       rel_doc_text=known_doc_text,
                                                       unk_doc_texts=unk_doc_texts)

        # TODO: Remove safety check
        if len(unk_doc_ids) != len(inferred_scores):
            print("Error: Inferred scores mismatch")
            exit()

        for unk_doc_id, score in zip(unk_doc_ids, inferred_scores):
            unk_doc_scores[unk_doc_id] = score

        # Write the scores to the cache
        with gzip.open(DUOPROMPT_CACHE, 'at') as file:
            for unk_doc_id, score in unk_doc_scores.items():
                file.write(json.dumps({"qid": qid,
                                       "known_passage_id": known_passage_id,
                                       "passage_to_judge_id": unk_doc_id,
                                       "score": score}) + '\n')

                # Add the scores to the local instance of the cache
                key = get_key([qid, known_passage_id, unk_doc_id])
                candidates_cache[key] = score

    print(f"Used cached scores: {used_cached_count}")
    print(f"Infered scores: {infered_count}")

    # Create the pairwise preferences file for the candidates
    with gzip.open(pairwise_preferences_path, 'wt') as file:
        # Write all candidates with known relevant passage to the file
        for key, group in grouped_relevant_candidates.items():
            qid, known_passage_id = key.split(KEY_SEPARATOR)
            for candidate in group:
                if candidate['known_relevant_passage']:
                    key = get_key([candidate['qid'],
                                   candidate['known_relevant_passage']['docno'],
                                   candidate['passage_to_judge']['docno']])
                elif candidate['known_non_relevant_passage']:
                    key = get_key([candidate['qid'],
                                   candidate['known_non_relevant_passage']['docno'],
                                   candidate['passage_to_judge']['docno']])
                else:
                    print("Error: Candidate is neither known relevant nor known non relevant")
                    exit()

                file.write(json.dumps({"qid": candidate['qid'],
                                       "known_relevant_passage_id": candidate['known_relevant_passage']['docno'],
                                       "known_non_relevant_passage_id": "",  # False
                                       "passage_to_judge_id": candidate['passage_to_judge']['docno'],
                                       "score": candidates_cache[key]}) + '\n')

        # Write all candidates with known non relevant passage to the file
        for key, group in grouped_non_relevant_candidates.items():
            qid, known_passage_id = key.split(KEY_SEPARATOR)
            for candidate in group:
                if candidate['known_relevant_passage']:
                    key = get_key([candidate['qid'],
                                   candidate['known_relevant_passage']['docno'],
                                   candidate['passage_to_judge']['docno']])
                elif candidate['known_non_relevant_passage']:
                    key = get_key([candidate['qid'],
                                   candidate['known_non_relevant_passage']['docno'],
                                   candidate['passage_to_judge']['docno']])
                else:
                    print("Error: Candidate is neither known relevant nor known non relevant")
                    exit()

                file.write(json.dumps({"qid": candidate['qid'],
                                       "known_relevant_passage_id": "",  # False
                                       "known_non_relevant_passage_id": candidate['known_non_relevant_passage']['docno'],
                                       "passage_to_judge_id": candidate['passage_to_judge']['docno'],
                                       "score": candidates_cache[key]}) + '\n')


if __name__ == '__main__':

    for candidates_path in glob(FILE_PATTERN):

        # Extract the file name
        file_name = os.path.basename(candidates_path)
        pairwise_preferences_path = os.path.join(DUOPROMPT_PATH, file_name)

        if os.path.exists(pairwise_preferences_path):
            print(f"Already processed {pairwise_preferences_path}")
            continue

        start_time = time.time()
        print(f"Processing {candidates_path}")
        process_candidates(candidates_path, pairwise_preferences_path)
        print(f"Processed {file_name} in {(time.time() - start_time) / 60} minutes")
