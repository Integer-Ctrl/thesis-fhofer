#!/usr/bin/env python3
import gzip
import ir_datasets
import json
import os


SELECTED_QUERIES = {
    'msmarco-passage/trec-dl-2019/judged': ('1037798', '1129237'),
    'msmarco-passage/trec-dl-2020/judged': ('997622', '1051399', '1127540'),
    'argsme/2020-04-01/touche-2020-task-1': ('49', '34'),
    'disks45/nocr/trec-robust-2004': ('681', '448'),
    'disks45/nocr/trec7': ('354', '358'),
    'disks45/nocr/trec8': ('441', '422')
}

DATA_PATH = '/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-fhofer/data'
CLUEWEB_PATH = '/clueweb22/b/candidates-chatnoir'
APPROACH = '/union_50_opd.jsonl.gz'

CANDIDATE_PATHS = {
    'msmarco-passage/trec-dl-2019/judged': f'{DATA_PATH}/msmarco-passage/trec-dl-2019/judged{CLUEWEB_PATH}{APPROACH}',
    'msmarco-passage/trec-dl-2020/judged': f'{DATA_PATH}/msmarco-passage/trec-dl-2020/judged{CLUEWEB_PATH}{APPROACH}',
    'argsme/2020-04-01/touche-2020-task-1': f'{DATA_PATH}/argsme/2020-04-01/touche-2020-task-1{CLUEWEB_PATH}{APPROACH}',
    'disks45/nocr/trec-robust-2004': f'{DATA_PATH}/disks45/nocr/trec-robust-2004{CLUEWEB_PATH}{APPROACH}',
    'disks45/nocr/trec7': f'{DATA_PATH}/disks45/nocr/trec7{CLUEWEB_PATH}{APPROACH}',
    'disks45/nocr/trec8': f'{DATA_PATH}/disks45/nocr/trec8{CLUEWEB_PATH}{APPROACH}',
}


def print_selected_queries():
    for dataset_id, qids in SELECTED_QUERIES.items():
        dataset = ir_datasets.load(dataset_id)
        for q in dataset.queries_iter():
            if str(q.query_id) in qids:
                print(q)


def write_docs_to_judge():
    for dataset_id, qids in SELECTED_QUERIES.items():

        print(f'Processing: {dataset_id}')

        CANDIDATE_PATH = f'{DATA_PATH}/{dataset_id}{CLUEWEB_PATH}{APPROACH}'
        DOC_IDS_PATHS = f'{DATA_PATH}/{dataset_id}{CLUEWEB_PATH}/doc_ids.jsonl'

        if not os.path.exists(CANDIDATE_PATH):
            print(f'Skipping: {dataset_id} because candidate path does not exist')
            continue

        doc_ids = {}
        for qid in qids:
            doc_ids[qid] = set()
        
        with gzip.open(CANDIDATE_PATH, 'rt', encoding='UTF-8') as file:
            for line in file:
                line = json.loads(line)
                qid = line['qid']
                if qid not in qids:
                    print(f'ERROR: qid {qid} not in selected queries')
                    exit()

                passage_no = line['passage_to_judge']['docno']
                docno = passage_no.split('___')[0]
                doc_ids[qid].add(docno)

        with open(DOC_IDS_PATHS, 'wt') as file:
            for qid, doc_ids in doc_ids.items():
                file.write(json.dumps({'qid': qid, 'doc_ids': list(doc_ids)}) + '\n')
        

if __name__ == '__main__':

    write_docs_to_judge()        
