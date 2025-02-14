import gzip
import json
import pyterrier as pt
import os
from glob import glob


separator = '---'
dataset_order = ['touche-2020-task-1', 'trec-robust-2004', 'trec7', 'trec8', 'trec-dl-2019/judged', 'trec-dl-2020/judged']
aggre_order = ['mean', 'min', 'max', 'sum']
eval_order = ['kendall', 'spearman']
tra_order = ['id', 'log', 'exp', 'sqrt']
paths = {  # used candidate as key : path
    'nearest_neighbor_10.jsonl.gz': '/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-fhofer/data/argsme/2020-04-01/touche-2020-task-1/argsme/2020-04-01/touche-2020-task-1/duoprompt/',
    # '/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-fhofer/data/disks45/nocr/trec7/disks45/nocr/trec7/monoprompt/',
    # '/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-fhofer/data/disks45/nocr/trec8/disks45/nocr/trec8/monoprompt/',
    # '/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-fhofer/data/disks45/nocr/trec-robust-2004/disks45/nocr/trec-robust-2004/monoprompt/',
    # '/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-fhofer/data/msmarco-document/trec-dl-2019/judged/msmarco-document/trec-dl-2019/judged/monoprompt/',
    # '/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-fhofer/data/msmarco-document/trec-dl-2020/judged/msmarco-document/trec-dl-2020/judged/monoprompt',
}

# Load the configuration settings
pwd = os.path.dirname(os.path.abspath(__file__))


def load_config(filename=pwd + "/../../config.json"):
    with open(filename, "r") as f:
        config = json.load(f)
    return config


# Get the configuration settings
config = load_config()

DOCUMENT_DATASET_TARGET_NAME = config['DOCUMENT_DATASET_TARGET_NAME']
DOCUMENT_DATASET_TARGET_NAME_PYTERRIER = config['DOCUMENT_DATASET_TARGET_NAME_PYTERRIER']

SOURCE_PATH = os.path.join(config['DATA_PATH'], config["DOCUMENT_DATASET_SOURCE_NAME"])
TARGET_PATH = os.path.join(SOURCE_PATH, config["DOCUMENT_DATASET_TARGET_NAME"])

EVALUATION_METHODS = config['EVALUATION_METHODS']
NUMBER_OF_CROSS_VALIDATION_FOLDS = config['NUMBER_OF_CROSS_VALIDATION_FOLDS']
KEY_SEPARATOR = config['KEY_SEPARATOR']

BACKBONES = config['BACKBONES']  # all backbones
MONOPROMPT_PATH = config['MONOPROMPT_PATH']


for backbone in BACKBONES:

    print(backbone)

    for APPROACH, PATH in paths.items():

        path = os.path.join(PATH, backbone, config['LABEL_CORRELATION_AVR_PER_QUERY_PATH'], APPROACH)

        correlation_scores = {}

        with gzip.open(path, 'rt', encoding='UTF-8') as file:
            for line in file:
                line = json.loads(line)
                eval_method, aggre_met, tra_met, _, _ = line['evaluation_method'].split('---')

                if aggre_met not in correlation_scores:
                    correlation_scores[aggre_met] = {}
                if tra_met not in correlation_scores[aggre_met]:
                    correlation_scores[aggre_met][tra_met] = {}
                if eval_method not in correlation_scores[aggre_met][tra_met]:
                    correlation_scores[aggre_met][tra_met][eval_method] = line['score']

        for aggre_met in aggre_order:
            for tra_met in tra_order:
                for eval_method in eval_order:
                    print(correlation_scores[aggre_met][tra_met][eval_method])
