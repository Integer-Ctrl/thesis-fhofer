import os
import gzip
import json

separator = '---'
# dataset_order = ['touche-2020-task-1', 'trec-robust-2004', 'trec7', 'trec8', 'trec-dl-2019/judged', 'trec-dl-2020/judged']
dataset_order = ['touche-2020-task-1', 'trec7', 'trec8', 'trec-dl-2019/judged', 'trec-dl-2020/judged']
approach_order = ['naive', 'naive_opd',
                  'nearest_neighbor_10', 'nearest_neighbor_50', 'nearest_neighbor_100',
                  'nearest_neighbor_10_opd', 'nearest_neighbor_50_opd', 'nearest_neighbor_100_opd',
                  'union_10', 'union_50', 'union_100',
                  'union_10_opd', 'union_50_opd', 'union_100_opd',]

paths = [
    '/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-fhofer/data/argsme/2020-04-01/touche-2020-task-1/argsme/2020-04-01/touche-2020-task-1/candidates/',
    '/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-fhofer/data/disks45/nocr/trec7/disks45/nocr/trec7/candidates/',
    '/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-fhofer/data/disks45/nocr/trec8/disks45/nocr/trec8/candidates/',
    # '/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-fhofer/data/disks45/nocr/trec-robust-2004/disks45/nocr/trec-robust-2004/candidates/',
    '/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-fhofer/data/msmarco-document/trec-dl-2019/judged/msmarco-document/trec-dl-2019/judged/candidates/',
    '/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-fhofer/data/msmarco-document/trec-dl-2020/judged/msmarco-document/trec-dl-2020/judged/candidates',
]


def print_table():
    recall = {}
    docs = {}

    for approach in approach_order:
        recall[approach] = {}
        docs[approach] = {}

    for path in paths:

        dataset = next(ds for ds in dataset_order if ds in path)

        file_path = os.path.join(path, 'results.json')
        with open(file_path, 'rt') as f:
            data = json.load(f)

            for approach in approach_order:
                recall[approach][dataset] = data[approach]['relevant_recall']
                docs[approach][dataset] = data[approach]['num_candidates']

    for approach in approach_order:
        print('&', end=' ')
        for dataset in dataset_order:
            print(f'& {round(recall[approach][dataset], 3)} & {docs[approach][dataset]}', end=' ')
        print('\\\\')


print('Candidate retrieval evaluation')
print_table()
