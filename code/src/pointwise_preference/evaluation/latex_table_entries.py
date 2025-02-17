import gzip
import json
import pyterrier as pt
import os
from glob import glob


separator = '---'
aggre_order = ['max']
eval_order = ['kendall', 'spearman']
tra_order = ['id']

DATA_PATH = '/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-fhofer/data'
PATHS = [
    'argsme/2020-04-01/touche-2020-task-1',
    # 'disks45/nocr/trec-robust-2004',
    'disks45/nocr/trec7',
    'disks45/nocr/trec8',
    'msmarco-document/trec-dl-2019/judged',
    'msmarco-document/trec-dl-2020/judged',
]
BACKBONES =  ['google/flan-t5-base',
              'google/flan-t5-small',
              'google-t5/t5-small']
AVR_PATH = 'avr-per-query'
APPROACH = 'union_50_opd.jsonl.gz'

correlation_scores = {}

for PATH in PATHS:

    if PATH not in correlation_scores:
        correlation_scores[PATH] = {}

    for BACKBONE in BACKBONES:

        if BACKBONE not in correlation_scores[PATH]:
            correlation_scores[PATH][BACKBONE] = {}
    
        path = f'{DATA_PATH}/{PATH}/{PATH}/duoprompt/{BACKBONE}/{AVR_PATH}/{APPROACH}'
        print(path)

        with gzip.open(path, 'rt', encoding='UTF-8') as file:
            for line in file:
                line = json.loads(line)

                eval_method, aggre_met, tra_met, _, _ = line['evaluation_method'].split('---')

                if aggre_met not in correlation_scores[PATH][BACKBONE]:
                    correlation_scores[PATH][BACKBONE][aggre_met] = {}
                if tra_met not in correlation_scores[PATH][BACKBONE][aggre_met]:
                    correlation_scores[PATH][BACKBONE][aggre_met][tra_met] = {}
                if eval_method not in correlation_scores[PATH][BACKBONE][aggre_met][tra_met]:
                    correlation_scores[PATH][BACKBONE][aggre_met][tra_met][eval_method] = line['score']


# default metrics
print('Default metrics')
for backbone in BACKBONES:
    print(backbone)
    for aggre_met in aggre_order:
        for tra_met in tra_order:
            print(f'& \multicolumn{{2}}{{c}}{{\\textbf{{{backbone}}}}}', end=' ')
            for PATH in PATHS:
                kendall = correlation_scores[PATH][backbone][aggre_met][tra_met]['kendall']
                spearman = correlation_scores[PATH][backbone][aggre_met][tra_met]['spearman']
                print(f'& {round(kendall, 3)} & {round(spearman, 3)}', end=' ')
            print('\\\\')

# greedy metrics
# print('Greedy metrics')
# for backbone in BACKBONES:
#     print(backbone)
#     for aggre_met in aggre_order:
#         print(f'& \\multirow{{4}}{{*}}{{\\textbf{{{aggre_met}}}}}')
#         for tra_met in tra_order:
#             print(f'\t& & \\textbf{{{tra_met}}}', end=' ')
#             for PATH in PATHS:
#                 kendall = correlation_scores[PATH][backbone][aggre_met][tra_met]['kendall-greedy']
#                 spearman = correlation_scores[PATH][backbone][aggre_met][tra_met]['spearman-greedy']
#                 print(f'& {round(kendall, 3)} & {round(spearman, 3)}', end=' ')
#             print('\\\\')
