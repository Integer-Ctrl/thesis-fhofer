import os
import gzip
import json

separator = '---'
dataset_order = ['touche-2020-task-1', 'trec-robust-2004', 'trec7', 'trec8', 'trec-dl-2019/judged', 'trec-dl-2020/judged']
dataset_num_queries = {
    'touche-2020-task-1': 49,
    'trec-robust-2004': 249,  # qid 672(249) and MAYBE 347(248)
    'trec7': 50,
    'trec8': 50,
    'trec-dl-2019/judged': 43,
    'trec-dl-2020/judged': 54,
}
approach_order = ['naive', 'naive_opd',
                  'nearest_neighbor_10', 'nearest_neighbor_50', 'nearest_neighbor_100',
                  'nearest_neighbor_10_opd', 'nearest_neighbor_50_opd', 'nearest_neighbor_100_opd',
                  'union_10', 'union_50', 'union_100',
                  'union_10_opd', 'union_50_opd', 'union_100_opd',]

paths = [
    '/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-fhofer/data/argsme/2020-04-01/touche-2020-task-1/argsme/2020-04-01/touche-2020-task-1/candidates/',
    '/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-fhofer/data/disks45/nocr/trec7/disks45/nocr/trec7/candidates/',
    '/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-fhofer/data/disks45/nocr/trec8/disks45/nocr/trec8/candidates/',
    '/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-fhofer/data/disks45/nocr/trec-robust-2004/disks45/nocr/trec-robust-2004/candidates/',
    '/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-fhofer/data/msmarco-passage/trec-dl-2019/judged/msmarco-passage/trec-dl-2019/judged/candidates/',
    '/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-fhofer/data/msmarco-passage/trec-dl-2020/judged/msmarco-passage/trec-dl-2020/judged/candidates',
]


def print_table():
    recall = {}
    docs = {}

    for approach in approach_order:
        recall[approach] = {}
        docs[approach] = {}

    for path in paths:

        dataset = next(ds for ds in dataset_order if ds in path)

        #### REMOVE WHEN ROBUST04 IS DONE ####
        if 'trec-robust-2004' in dataset:
            for approach in approach_order:
                recall[approach][dataset] = 0.000
                docs[approach][dataset] = 0.000
            continue
        #### REMOVE WHEN ROBUST04 IS DONE ####

        file_path = os.path.join(path, 'results.json')
        with open(file_path, 'rt') as f:
            data = json.load(f)

            for approach in approach_order:
                recall[approach][dataset] = data[approach]['relevant_recall']
                docs[approach][dataset] = data[approach]['num_candidates']

    for approach in approach_order:
        print('&', end=' ')
        for dataset in dataset_order:
            print(f'& ${round(recall[approach][dataset], 3)}$ & ${(docs[approach][dataset] // dataset_num_queries[dataset])}$', end=' ')
        print('\\\\')


print('Candidate retrieval evaluation')
print('Use regex for format: ([0-9]{1,3})([0-9]{3}) -> $1\\,$2')
print_table()
