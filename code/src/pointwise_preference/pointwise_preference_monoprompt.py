import autoqrels
import gzip
import json
import os
import autoqrels.zeroshot
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
CANDIDATES_FILE_PATTERN = os.path.join(CANDIDATES_PATH, "*.jsonl.gz")

ONLY_JUDGED = config['ONLY_JUDGED']  # only infer the scores for the judged documents
PREFERENCE_BACKBONE = config['PREFERENCE_BACKBONE']
print(f"Preference backbone: {PREFERENCE_BACKBONE}")
MONOPROMPT_PATH = os.path.join(TARGET_PATH, config['MONOPROMPT_PATH'], PREFERENCE_BACKBONE)
MONOPROMPT_CACHE = os.path.join(MONOPROMPT_PATH, config['MONOPROMPT_CACHE_NAME'])

if not os.path.exists(MONOPROMPT_PATH):
    os.makedirs(MONOPROMPT_PATH)

KEY_SEPARATOR = config['KEY_SEPARATOR']
PASSAGE_ID_SEPARATOR = config['PASSAGE_ID_SEPARATOR']


# Helper function to access the cache
def get_key(list):
    return KEY_SEPARATOR.join(list)


# Check if cache exists and load it if it does
# Key: qid---known_passage_id---upassage_to_judge_id: score
# Known passage is either known relevant or known non relevant
candidates_cache = {}
if os.path.exists(MONOPROMPT_CACHE):
    with gzip.open(MONOPROMPT_CACHE, 'rt') as file:
        for line in file:
            line = json.loads(line)

            key = get_key([line['qid'], line['passage_to_judge_id']])
            candidates_cache[key] = line['score']


def process_candidates(candidates_path, pointwise_preferences_path, judged_doc_ids):

    # Load the dataset and the MonoPrompt model
    dataset = ir_datasets.load(DOCUMENT_DATASET_TARGET_NAME_PYTHON_API)
    monoprompt = autoqrels.zeroshot.GradedMonoPrompt(dataset=dataset,
                                                     backbone=PREFERENCE_BACKBONE,
                                                     device='cuda',
                                                     batch_size=64)

    used_cached_count = 0
    infered_count = 0

    grouped_candidates = {}

    with gzip.open(candidates_path, 'rt') as file:
        for line in file:
            # Add all candidates to the list, also those that are already in the cache due to the cache is overwritten
            candidate = json.loads(line)
            qid = candidate['qid']

            if ONLY_JUDGED:
                docno = candidate['passage_to_judge']['docno'].split(PASSAGE_ID_SEPARATOR)[0]
                if docno not in judged_doc_ids[qid]:
                    continue

            if qid not in grouped_candidates:
                grouped_candidates[qid] = []

            # filter out multiple preferences (pairwise has 15 relevant and 5 non-relevant)
            if candidate['passage_to_judge']['docno'] in [x['passage_to_judge']['docno'] for x in grouped_candidates[qid]]:
                continue
            grouped_candidates[qid].append(candidate)

    # Check if candidates are in cache already and if not infer them
    # Iterate over the grouped candidates and infer the relevance
    for qid, candidates in tqdm(grouped_candidates.items(), desc="Infer relevance"):

        query_text = candidates[0]['query_text']
        if query_text == "":
            print("Error: Empty query text")
            exit()

        unk_doc_scores = {}
        unk_doc_ids = []
        unk_doc_texts = []

        for candidate in candidates:

            # TODO: Remove safety check
            if candidate['qid'] != qid:
                print("Error: qid mismatch")
                exit()

            # If the score is already in the cache, dont infer it again
            key = get_key([candidate['qid'],
                           candidate['passage_to_judge']['docno']])

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
        inferred_scores = monoprompt.infer_zeroshot_text(query_text=query_text,
                                                         unk_doc_texts=unk_doc_texts)

        # TODO: Remove safety check
        if len(unk_doc_ids) != len(inferred_scores):
            print("Error: Inferred scores mismatch")
            exit()

        for unk_doc_id, score in zip(unk_doc_ids, inferred_scores):
            unk_doc_scores[unk_doc_id] = score

        # Write the scores to the cache
        with gzip.open(MONOPROMPT_CACHE, 'at') as file:
            for unk_doc_id, score in unk_doc_scores.items():
                file.write(json.dumps({"qid": qid,
                                       "passage_to_judge_id": unk_doc_id,
                                       "score": score}) + '\n')

                # Add the scores to the local instance of the cache
                key = get_key([qid, unk_doc_id])
                candidates_cache[key] = score

    print(f"Used cached scores: {used_cached_count}")
    print(f"Infered scores: {infered_count}")

    # Create the pointwise preferences file for the candidates
    with gzip.open(pointwise_preferences_path, 'wt') as file:
        # Write all candidates with known relevant passage to the file
        for qid, candidates in grouped_candidates.items():
            for candidate in candidates:

                passage_to_judge_id = candidate['passage_to_judge']['docno']
                key = get_key([qid,
                               passage_to_judge_id])

                file.write(json.dumps({"qid": qid,
                                       "passage_to_judge_id": passage_to_judge_id,
                                       "score": candidates_cache[key]}) + '\n')


# Get the judged documents if the ONLY_JUDGED flag is set
def get_judged_doc_ids():
    judged_doc_ids = {}
    dataset = ir_datasets.load(DOCUMENT_DATASET_TARGET_NAME_PYTHON_API)

    for qrel in dataset.qrels_iter():
        qid = qrel.query_id
        if qid not in judged_doc_ids:
            judged_doc_ids[qid] = set()
        judged_doc_ids[qid].add(qrel.doc_id)

    return judged_doc_ids


if __name__ == '__main__':

    for candidates_path in glob(CANDIDATES_FILE_PATTERN):

        # Get judged document ids if the flag is set
        judged_doc_ids = None  # Set to None if all documents should be inferred
        if ONLY_JUDGED:
            judged_doc_ids = get_judged_doc_ids()

        # Extract the file name
        file_name = os.path.basename(candidates_path)
        pointwise_preferences_path = os.path.join(MONOPROMPT_PATH, file_name)

        if os.path.exists(pointwise_preferences_path):
            print(f"Already processed {pointwise_preferences_path}")
            continue

        start_time = time.time()
        print(f"Processing {candidates_path}")
        process_candidates(candidates_path, pointwise_preferences_path, judged_doc_ids)
        print(f"Processed {file_name} in {(time.time() - start_time) / 60} minutes")
