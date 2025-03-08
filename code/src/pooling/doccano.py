import gzip
import json
import os
import numpy as np
import ir_datasets

SELECTED_QUERIES = {
    'msmarco-passage/trec-dl-2019/judged': ('1037798', '1129237'),
    'msmarco-passage/trec-dl-2020/judged': ('997622', '1051399', '1127540'),
    'argsme/2020-04-01/touche-2020-task-1': ('49', '34'),
    'disks45/nocr/trec-robust-2004': ('681', '448'),
    'disks45/nocr/trec7': ('354', '358'),
    'disks45/nocr/trec8': ('441', '422')
}

DATA_PATH = '/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-fhofer/data'
POOL_PATH = os.path.join(DATA_PATH, 'clueweb22-transfer', 'judgment-pool-10.json')
DOCCANO_PATH = os.path.join(DATA_PATH, 'clueweb22-transfer', 'doccano')
CW22_PATH = 'clueweb22/b'
BACKBONE = "google/flan-t5-base"
CANDIDATE_APPROACH = 'eval_candidates.jsonl.gz'
BEST_PASSAGE_NUM = 3


def write_doccano(dataset_name, query_id):
    print(f"Processing dataset {dataset_name} and query {query_id}")

    query_text = ''
    query_narrative = ''
    query_description = ''
    dataset = ir_datasets.load(dataset_name)
    for query in dataset.queries_iter():
        if query.query_id == query_id:
            query_text = query.default_text()
            query_narrative = ''
            if 'narrative' in dir(query):
                query_narrative = query.narrative
            query_description = ''
            if 'description' in dir(query):
                query_description = query.description
            break

    '''
    Load pool of candidates
    '''
    with open(POOL_PATH, 'r') as file:
        data = json.load(file)
        pool_ids = set(data[query_id])

    '''
    Load candidates with passage to judge text
    '''
    candidates_path = os.path.join(DATA_PATH, dataset_name, CW22_PATH, 'candidates-chatnoir', CANDIDATE_APPROACH)
    target_passages_text = {}


    with gzip.open(candidates_path, 'rt') as file:
        for line in file:
            data = json.loads(line)
            qid = data['qid']

            if qid != query_id:
                continue

            passage_to_judge_id = data['passage_to_judge']['docno']
            passage_to_judge_text = data['passage_to_judge']['text']
            document_to_judge_id = passage_to_judge_id.split('___')[0]

            if qid not in target_passages_text:
                target_passages_text[qid] = {}
            if document_to_judge_id not in target_passages_text[qid]:
                target_passages_text[qid][document_to_judge_id] = {}
            if passage_to_judge_id not in target_passages_text[qid][document_to_judge_id]:
                target_passages_text[qid][document_to_judge_id][passage_to_judge_id] = passage_to_judge_text

    '''
    Create doccano entries for best backbone
    '''
    duoprompt_path = os.path.join(DATA_PATH, dataset_name, CW22_PATH, 'duoprompt', BACKBONE, CANDIDATE_APPROACH)

    if not os.path.exists(duoprompt_path) or not os.path.exists(candidates_path):
        print(f"Paths do not exist: {duoprompt_path}, {candidates_path}")
        exit()

    '''
    Load the inferred passage scores from the duoprompt and monoprompt for backbone
    '''
    inferred_passage_scores_duoprompt = {}
    inferred_passage_scores_monoprompt = {}

    # PAIRWISE
    with gzip.open(duoprompt_path, 'rt') as file:
        for line in file:
            data = json.loads(line)
            qid = data['qid']
            passage_to_judge_id = data['passage_to_judge_id']
            document_to_judge_id = passage_to_judge_id.split('___')[0]
            score = data['score']

            if qid not in inferred_passage_scores_duoprompt:
                inferred_passage_scores_duoprompt[qid] = {}
            if document_to_judge_id not in inferred_passage_scores_duoprompt[qid]:
                inferred_passage_scores_duoprompt[qid][document_to_judge_id] = {}
            if passage_to_judge_id not in inferred_passage_scores_duoprompt[qid][document_to_judge_id]:
                inferred_passage_scores_duoprompt[qid][document_to_judge_id][passage_to_judge_id] = []
            inferred_passage_scores_duoprompt[qid][document_to_judge_id][passage_to_judge_id].append(score)  # multiple scores for the same passage
            

    # pairwise: min aggregation and id transformaiton on passage level
    for qid, documents in inferred_passage_scores_duoprompt.items():
        for document_id, passages in documents.items():
            for passage_id, scores in passages.items():
                inferred_passage_scores_duoprompt[qid][document_id][passage_id] = float(np.min(scores))
    
    '''
    Create doccano input file for the best BEST_PASSAGE_NUM passages
    '''
    # PAIRWISE
    doccano_input = []
    found_doc_ids = 0
    not_found_doc_ids = 0

    for doc_id in pool_ids:
        documents = inferred_passage_scores_duoprompt[query_id]

        if doc_id not in documents:
            not_found_doc_ids += 1
            continue
        found_doc_ids += 1

        passages = documents[doc_id]
        document_best_passages = sorted(passages.items(), key=lambda x: x[1], reverse=True)[:BEST_PASSAGE_NUM]
        url = f'https://chatnoir-webcontent.web.webis.de/?index=cw22&trec-id={doc_id}'

        for passage_id, score in document_best_passages:
            doccano_entry = {
                'label': [],
                'text': target_passages_text[query_id][doc_id][passage_id],
                'query': query_text,
                'query_id': query_id,
                'query_narrative': query_narrative,
                'query_description': query_description,
                'document_id': doc_id,
                'passage_id': passage_id,
                'url': url,
            }
            doccano_input.append(doccano_entry)
    
    print(f"Found {found_doc_ids} documents and {not_found_doc_ids} documents not found")
    return doccano_input


for dataset_name, query_ids in SELECTED_QUERIES.items():
    for query_id in query_ids:
        doccano_data = write_doccano(dataset_name, query_id)

        with open(os.path.join(DOCCANO_PATH, f'doccano_{query_id}.jsonl'), 'w') as file:
            for entry in doccano_data:
                file.write(json.dumps(entry) + '\n')
