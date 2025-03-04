import os
import gzip
import json

separator = '---'
dataset_order = ['touche-2020-task-1', 'trec-robust-2004', 'trec7', 'trec8', 'trec-dl-2019/judged', 'trec-dl-2020/judged']
metrics_order = ['ndcg10', 'p10']
retriever_order = ['BM25', 'DFR_BM25', 'DFIZ', 'DLH', 'DPH', 'DirichletLM', 'Hiemstra_LM', 'LGD', 'PL2', 'TF_IDF']
metric_retriever_combinations = [f'{metric}_{retriever}' for metric in metrics_order for retriever in retriever_order]

paths = [
    '/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-fhofer/data/argsme/2020-04-01/touche-2020-task-1/',
    '/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-fhofer/data/disks45/nocr/trec-robust-2004/',
    '/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-fhofer/data/disks45/nocr/trec7/',
    '/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-fhofer/data/disks45/nocr/trec8/',
    '/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-fhofer/data/msmarco-passage/trec-dl-2019/judged/',
    '/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-fhofer/data/msmarco-passage/trec-dl-2020/judged/',
]


def print_table(greedy):

    kendall_default = {}
    spearman_default = {}

    for retriever in retriever_order:
        kendall_default[retriever] = {}
        spearman_default[retriever] = {}
        for metric in metrics_order:
            kendall_default[retriever][metric] = {}
            spearman_default[retriever][metric] = {}

    for path in paths:

        dataset = next(ds for ds in dataset_order if ds in path)

        # #### REMOVE WHEN ROBUST04 IS DONE ####
        # if 'trec-robust-2004' in dataset:
        #     for retriever in retriever_order:
        #         for metric in metrics_order:
        #             kendall_default[retriever][metric][dataset] = 0.000
        #             spearman_default[retriever][metric][dataset] = 0.000
        #     continue
        # #### REMOVE WHEN ROBUST04 IS DONE ####

        file_path = os.path.join(path, 'avg-rank-correlation-scores.jsonl.gz')
        with gzip.open(file_path, 'rt') as f:
            for line in f:
                data = json.loads(line)

                score = data['score']
                eval___aggr___tra___retriever___metric = data['eval---aggr---tra---retriever---metric']  # metric e.g. ndcg10_DirichletLM
                eval_method, aggr_method, tra_method, retriever, metric = eval___aggr___tra___retriever___metric.split('---')

                if greedy:
                    if eval_method == 'kendall-greedy':
                        kendall_default[retriever][metric][dataset] = score
                    if eval_method == 'spearman-greedy':
                        spearman_default[retriever][metric][dataset] = score
                else:
                    if eval_method == 'kendall':
                        kendall_default[retriever][metric][dataset] = score
                    if eval_method == 'spearman':
                        spearman_default[retriever][metric][dataset] = score

    for metric in metrics_order:
        print(f"\multirow{{10}}{{*}}{{\\rotatebox{{90}}{{\\texttt{{{metric}}}}}}}")
        for retriever in retriever_order:
            kendall_sum = 0.0
            spearman_sum = 0.0

            print(f"\t& {retriever}", end=' ')
            for dataset in dataset_order:
                kendall_sum += kendall_default[retriever][metric][dataset]
                spearman_sum += spearman_default[retriever][metric][dataset]

                print(f'& {round(kendall_default[retriever][metric][dataset], 3)} & {round(spearman_default[retriever][metric][dataset], 3)}', end=' ')
            print(f'& {round((kendall_sum/len(dataset_order)), 3)} & {round((spearman_sum/len(dataset_order)), 3)} \\\\')


print('Default rank correlation')
print_table(greedy=False)

print('Greedy rank correlation')
print_table(greedy=True)
