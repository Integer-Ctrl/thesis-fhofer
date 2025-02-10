#!/usr/bin/env python3
import ir_datasets


SELECTED_QUERIES = {
    'msmarco-passage/trec-dl-2019/judged': ('1037798', '1129237'),
    'msmarco-passage/trec-dl-2020/judged': ('997622', '1051399', '1127540'),
    'argsme/2020-04-01/touche-2020-task-1': ('49', '34'),
    'disks45/nocr/trec-robust-2004': ('681', '448'),
    'disks45/nocr/trec7': ('354', '358'),
    'disks45/nocr/trec8': ('441', '422')
}


for dataset_id, qids in SELECTED_QUERIES.items():
    dataset = ir_datasets.load(dataset_id)
    for q in dataset.queries_iter():
        if str(q.query_id) in qids:
            print(q)

