import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import json
import gzip
import numpy as np

DATA_PATH = '/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-fhofer/data'
PATHS = [
    'argsme/2020-04-01/touche-2020-task-1',
    'disks45/nocr/trec-robust-2004',
    'disks45/nocr/trec7',
    'disks45/nocr/trec8',
    'msmarco-passage/trec-dl-2019/judged',
    'msmarco-passage/trec-dl-2020/judged',
]
BACKBONES =  ['google/flan-t5-base',
              'google/flan-t5-small',
              'google-t5/t5-small']

APPROACH = 'union_100_opd.jsonl.gz'
CW22_APPROACH = 'eval_candidates.jsonl.gz'

# Step 1: Load data once and store in a dictionary
data_dict = {}
global_max_y = 0

for backbone in BACKBONES:
    model_name = backbone.split('/')[1]
    data = []

    # Load data for this backbone
    for dataset in PATHS:
        path = f'{DATA_PATH}/{dataset}/{dataset}/monoprompt/{backbone}/{APPROACH}'  # self-transfer
        # path = f'{DATA_PATH}/{dataset}/clueweb22/b/monoprompt/{backbone}/{CW22_APPROACH}'  # cw22-transfer
        with gzip.open(path, 'rt', encoding='UTF-8') as f:
            for line in f:
                data.append(json.loads(line.strip()))

    # Store data in dictionary
    df = pd.DataFrame(data)
    data_dict[model_name] = df

    # Compute histogram max frequency
    counts, _ = np.histogram(df['score'], bins=50)
    global_max_y = max(global_max_y, max(counts))  # Update global max if necessary

# Step 2: Generate plots using preloaded data
for model_name, df in data_dict.items():

    # Define bin edges for the full range [0,1] with 50 bins
    bins = np.linspace(0, 1, 51)  # 51 edges define 50 bins
    # Create a histogram plot
    g = sns.displot(df, x='score', kde=False, bins=bins, height=6, aspect=1.5)

    # Set fixed x-axis and y-axis limit
    g.ax.set_xlim(0, 1)
    g.ax.set_ylim(0, global_max_y)
    
    # Set y-axis to scientific notation
    g.ax.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    g.ax.yaxis.get_offset_text().set_fontsize(14)

    # Labels and title
    g.ax.set_xlabel('Score', fontsize=24)
    g.ax.set_ylabel('Absolute Frequency', fontsize=24)
    # g.ax.set_title(f'Distribution of relevance scores generated using {model_name}', 
    #                fontsize=16, weight="bold", loc="center")
    g.ax.grid(True, linestyle="--", alpha=0.6)
    g.tick_params(labelsize=18)

    # Save the plot instead of showing it
    plt.savefig(f'pointwise_self_score_distribution_{model_name}.pdf', bbox_inches='tight')
    # plt.savefig(f'pointwise_cw22_score_distribution_{model_name}.pdf', bbox_inches='tight')
    plt.close()
