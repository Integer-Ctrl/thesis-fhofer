import gzip
import json
import os
import numpy as np

SELECTED_QUERIES = {
    'msmarco-passage/trec-dl-2019/judged': ('1037798', '1129237'),
    'msmarco-passage/trec-dl-2020/judged': ('997622', '1051399', '1127540'),
    'argsme/2020-04-01/touche-2020-task-1': ('49', '34'),
    'disks45/nocr/trec-robust-2004': ('681', '448'),
    'disks45/nocr/trec7': ('354', '358'),
    'disks45/nocr/trec8': ('441', '422')
}

DATA_PATH = '/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-fhofer/data'
CW22_PATH = 'clueweb22/b'
BACKBONES = ["google/flan-t5-base", "google/flan-t5-small", "google-t5/t5-small"]
CANDIDATE_APPROACH = 'union_100_opd.jsonl.gz'
# DOCCANO_PATH = os.path.join(DATA_PATH, 'clueweb22-transfer', 'doccano.jsonl')
DOCCANO_PATH = os.path.join(DATA_PATH, 'doccano.jsonl')
BEST_PASSAGE_NUM = 3

collection = {}

for dataset_name, query_ids in SELECTED_QUERIES.items():
    print(f"Processing dataset {dataset_name}")
    collection[dataset_name] = {}

    '''
    Load candidates with passage to judge text
    '''
    candidates_path = os.path.join(DATA_PATH, dataset_name, CW22_PATH, 'candidates-chatnoir', CANDIDATE_APPROACH)
    target_passages_text = {}


    with gzip.open(candidates_path, 'rt') as file:
        for line in file:
            data = json.loads(line)
            qid = data['qid']

            if qid not in query_ids:
                print(f"Skip query {qid}")
                exit()

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
    Create doccano entries for each tested backbone
    '''
    for backbone in BACKBONES:
        collection[dataset_name][backbone] = {}

        duoprompt_path = os.path.join(DATA_PATH, dataset_name, CW22_PATH, 'duoprompt', backbone, CANDIDATE_APPROACH)
        monoprompt_path = os.path.join(DATA_PATH, dataset_name, CW22_PATH, 'monoprompt', backbone, CANDIDATE_APPROACH)

        if not os.path.exists(duoprompt_path) or not os.path.exists(monoprompt_path) or not os.path.exists(candidates_path):
            print(f"Paths do not exist: {duoprompt_path}, {monoprompt_path}, {candidates_path}")
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
        
        # POINTWISE
        with gzip.open(monoprompt_path, 'rt') as file:
            for line in file:
                data = json.loads(line)
                qid = data['qid']
                passage_to_judge_id = data['passage_to_judge_id']
                document_to_judge_id = passage_to_judge_id.split('___')[0]
                score = data['score']

                if qid not in inferred_passage_scores_monoprompt:
                    inferred_passage_scores_monoprompt[qid] = {}
                if document_to_judge_id not in inferred_passage_scores_monoprompt[qid]:
                    inferred_passage_scores_monoprompt[qid][document_to_judge_id] = {}
                inferred_passage_scores_monoprompt[qid][document_to_judge_id][passage_to_judge_id] = score  # single score for each passage
        
        '''
        Create doccano input file for the best BEST_PASSAGE_NUM passages
        '''
        # PAIRWISE
        doccano_input_duoprompt = []
        for qid, documents in inferred_passage_scores_duoprompt.items():
            for document_id, passages in documents.items():
                document_best_passages = sorted(passages.items(), key=lambda x: x[1], reverse=True)[:BEST_PASSAGE_NUM]
                for passage_id, score in document_best_passages:
                    doccano_input_duoprompt.append({
                        'text': target_passages_text[qid][document_id][passage_id],
                        'meta': {
                            'qid': qid,
                            'docno': passage_id,
                            'score': score
                        }
                    })
        
        # POINTWISE
        doccano_input_monoprompt = []
        for qid, documents in inferred_passage_scores_monoprompt.items():
            for document_id, passages in documents.items():
                document_best_passages = sorted(passages.items(), key=lambda x: x[1], reverse=True)[:BEST_PASSAGE_NUM]
                for passage_id, score in document_best_passages:
                    doccano_input_monoprompt.append({
                        'text': target_passages_text[qid][document_id][passage_id],
                        'meta': {
                            'qid': qid,
                            'docno': passage_id,
                            'score': score
                        }
                    })
        
        collection[dataset_name][backbone] = {
            'duoprompt': doccano_input_duoprompt,
            'monoprompt': doccano_input_monoprompt
        }

print("Writing doccano file")
with open(DOCCANO_PATH, 'w') as file:
    json.dump(collection, file)
print("Finished writing doccano file")
