import os
import gzip
import json

separator = '---'
dataset_order = ['touche-2020-task-1', 'trec-robust-2004', 'trec7', 'trec8', 'trec-dl-2019/judged', 'trec-dl-2020/judged']
retriever_order = ['BM25', 'DFR_BM25', 'DFIZ', 'DLH', 'DPH', 'DirichletLM', 'Hiemstra_LM', 'LGD', 'PL2', 'TF_IDF']

paths = [
    '/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-fhofer/data/argsme/2020-04-01/touche-2020-task-1/',
    '/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-fhofer/data/disks45/nocr/trec-robust-2004/',
    '/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-fhofer/data/disks45/nocr/trec7/',
    '/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-fhofer/data/disks45/nocr/trec8/',
    '/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-fhofer/data/msmarco-document/trec-dl-2019/judged/',
    '/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-fhofer/data/msmarco-document/trec-dl-2020/judged/',
]


def print_table(greedy):
    kendall_default = {}
    spearman_default = {}

    for retriever in retriever_order:
        kendall_default[retriever] = {}
        spearman_default[retriever] = {}

    for path in paths:

        dataset = next(ds for ds in dataset_order if ds in path)

        file_path = os.path.join(path, 'cross-validation.jsonl.gz')
        with gzip.open(file_path, 'rt') as f:
            for line in f:
                data = json.loads(line)

                eval_method___retriever = data['eval_method___retriever']
                eval_method = eval_method___retriever.split(separator)[0]
                retriever = eval_method___retriever.split(separator)[1]
                score = data['score']

                if greedy:
                    if eval_method == 'kendall-greedy':
                        kendall_default[retriever][dataset] = score
                    if eval_method == 'spearman-greedy':
                        spearman_default[retriever][dataset] = score
                else:
                    if eval_method == 'kendall':
                        kendall_default[retriever][dataset] = score
                    if eval_method == 'spearman':
                        spearman_default[retriever][dataset] = score

    for retriever in retriever_order:
        print(retriever, end=' ')
        for dataset in dataset_order:
            print(f'& {round(kendall_default[retriever][dataset], 3)} & {round(spearman_default[retriever][dataset], 3)}', end=' ')
        print('\\\\')


print('Default rank correlation')
print_table(greedy=False)

print('Greedy rank correlation')
print_table(greedy=True)
