import os
import matplotlib.pyplot as plt
import pyterrier as pt
from glob import glob
import json
import gzip
import ray

ray.init()


@ray.remote
def ray_wrapper():
    def load_config(filename="/mnt/ceph/storage/data-tmp/current/ho62zoq/thesis-fhofer/code/src/config.json"):
        with open(filename, "r") as f:
            config = json.load(f)
        return config

    # Get the configuration settings
    config = load_config()

    # Either retrrieve with local index or with ChatNoir API
    CHATNOIR_RETRIEVAL = config['CHATNOIR_RETRIEVAL']

    DOCUMENT_DATASET_SOURCE_NAME = config['DOCUMENT_DATASET_SOURCE_NAME']
    DOCUMENT_DATASET_TARGET_NAME_PYTERRIER = config['DOCUMENT_DATASET_TARGET_NAME_PYTERRIER']

    SOURCE_PATH = os.path.join(config['DATA_PATH'], DOCUMENT_DATASET_SOURCE_NAME)
    TARGET_PATH = os.path.join(SOURCE_PATH, config["DOCUMENT_DATASET_TARGET_NAME"])

    if CHATNOIR_RETRIEVAL:
        CANDIDATES_PATH = os.path.join(TARGET_PATH, config['CANDIDATE_CHATNOIR_PATH'])
    else:
        CANDIDATES_PATH = os.path.join(TARGET_PATH, config['CANDIDATES_LOCAL_PATH'])
        FILE_PATTERN = os.path.join(CANDIDATES_PATH, "*.jsonl.gz")

    PASSAGE_ID_SEPARATOR = config['PASSAGE_ID_SEPARATOR']

    ##################
    # LOAD QREL FILE #
    ##################
    num_all_relevant_documents_per_query = {}
    num_all_judged_documents_per_query = {}

    dataset = pt.get_dataset(DOCUMENT_DATASET_TARGET_NAME_PYTERRIER)
    qrels = dataset.get_qrels(variant='relevance')
    for index, row in qrels.iterrows():
        # Count the number of judged documents per query
        if row['qid'] not in num_all_judged_documents_per_query:
            num_all_judged_documents_per_query[row['qid']] = 0
        num_all_judged_documents_per_query[row['qid']] += 1

        # Count the number of relevant documents per query
        if row['label'] > 0:
            if row['qid'] not in num_all_relevant_documents_per_query:
                num_all_relevant_documents_per_query[row['qid']] = 0
            num_all_relevant_documents_per_query[row['qid']] += 1

    ########################
    # EVALUATION FUNCTIONS #
    ########################

    # Function to plot Precision and Recall for each query and optionally save to PDF

    def plot_precision_recall(recalls, precisions, filename=None):
        # Check if a filename is provided
        if filename:
            # Prepare data
            queries = list(recalls.keys())
            recall_values = list(recalls.values())
            precision_values = list(precisions.values())

            # Create the plot
            plt.figure(figsize=(10, 8))
            plt.scatter(recall_values, precision_values, color='blue', s=100, alpha=0.7)

            # Annotate each point with its query ID
            for i, query in enumerate(queries):
                plt.text(recall_values[i], precision_values[i], query, fontsize=10, ha='right', va='bottom')

            # Set axis limits
            plt.xlim(0, 1)
            plt.ylim(0, 1)

            # Add labels and grid
            plt.title('Precision vs. Recall per Query', fontsize=16)
            plt.xlabel('Recall', fontsize=14)
            plt.ylabel('Precision', fontsize=14)
            plt.grid(True)

            # Tight layout for better spacing
            plt.tight_layout()

        # Save plot
            path = os.path.join(CANDIDATES_PATH, filename)
            plt.savefig(path, format='pdf')
            print(f"Plot saved to {filename}")

    # Compute Recall and Precision for each approach

    def compute_recall_precision(qid_docnos_cache, filename=None):

        num_retrieved_documents_per_query = {}
        # 1. Number of relevant documents per query
        num_retrieved_relevant_documents_per_query = {}
        # 2. Number of judged documents per query
        num_retrieved_judged_documents_per_query = {}

        for qid, docnos in qid_docnos_cache.items():
            num_retrieved_documents_per_query[qid] = len(docnos)
            num_retrieved_relevant_documents_per_query[qid] = 0
            num_retrieved_judged_documents_per_query[qid] = 0

            for docno in docnos:
                # Count the number of judged documents per query
                if docno in qrels[qrels['qid'] == qid]['docno'].values:
                    num_retrieved_judged_documents_per_query[qid] += 1

                # Count the number of relevant documents per query
                if docno in qrels[(qrels['qid'] == qid) & (qrels['label'] > 0)]['docno'].values:
                    num_retrieved_relevant_documents_per_query[qid] += 1

        # 3. RELEVANCE: Compute recall and precision for each query
        relevant_recalls = {qid: num_retrieved_relevant_documents_per_query[qid] / num_all_relevant_documents_per_query[qid]
                            for qid in num_all_relevant_documents_per_query.keys()}
        relevant_precisions = {qid: num_retrieved_relevant_documents_per_query[qid] / num_retrieved_documents_per_query[qid]
                               for qid in num_all_relevant_documents_per_query.keys()}

        # 4. JUDGEMENT: Compute recall and precision for each query
        judged_recalls = {qid: num_retrieved_judged_documents_per_query[qid] / num_all_judged_documents_per_query[qid]
                          for qid in num_all_judged_documents_per_query.keys()}
        judged_precisions = {qid: num_retrieved_judged_documents_per_query[qid] / num_retrieved_documents_per_query[qid]
                             for qid in num_all_judged_documents_per_query.keys()}

        # 3. Plot the precision and recall for each query and save the plot as a PDF
        if filename:
            relevant_filename = filename.replace('.pdf', '_relevant.pdf')
            judged_filename = filename.replace('.pdf', '_judged.pdf')
            plot_precision_recall(relevant_recalls, relevant_precisions, filename=relevant_filename)
            plot_precision_recall(judged_recalls, judged_precisions, filename=judged_filename)

        # 4. Compute average recall and precision
        relevant_recall = sum(relevant_recalls.values()) / len(relevant_recalls)
        relevant_precision = sum(relevant_precisions.values()) / len(relevant_precisions)

        judged_recall = sum(judged_recalls.values()) / len(judged_recalls)
        judged_precision = sum(judged_precisions.values()) / len(judged_precisions)

        return relevant_recall, relevant_precision, judged_recall, judged_precision

    def read_qid_docnos_cache(file):
        qid_docnos_cache = {}
        num_candidates = 0

        with gzip.open(file, 'rt') as f:
            for line in f:
                data = json.loads(line)
                qid = data['qid']
                passage_no = data['passage_to_judge']['docno']
                docno = passage_no.split(PASSAGE_ID_SEPARATOR)[0]

                if qid not in qid_docnos_cache:
                    qid_docnos_cache[qid] = {}
                if docno not in qid_docnos_cache[qid]:
                    qid_docnos_cache[qid][docno] = []
                    num_candidates += 1
                qid_docnos_cache[qid][docno].append(passage_no)

        return qid_docnos_cache, num_candidates

    ############################
    #           MAIN           #
    ############################
    result_path = os.path.join(CANDIDATES_PATH, 'plots')
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    results = {}

    # Evaluate the candidate retrieval approaches
    for file in glob(FILE_PATTERN):
        file_name = os.path.basename(file).replace('.jsonl.gz', '')

        print(f"Evaluating {file_name}")
        qid_docnos_cache, num_candidates = read_qid_docnos_cache(file)
        relevant_recall, relevant_precision, judged_recall, judged_precision = compute_recall_precision(
            qid_docnos_cache, filename=f"{file_name}.pdf")

        results[file_name] = {
            'num_candidates': num_candidates,
            'relevant_recall': relevant_recall,
            'relevant_precision': relevant_precision,
            'judged_recall': judged_recall,
            'judged_precision': judged_precision
        }

    # Save the results to a JSON file
    with open(os.path.join(CANDIDATES_PATH, 'results.json'), 'w') as f:
        json.dump(results, f, indent=4)
        print(f"Results saved to {result_path}/evaluation.json")


if __name__ == '__main__':

    NUM_WORKERS = 1

    futures = []
    for job_id in range(1, NUM_WORKERS + 1):
        futures.append(ray_wrapper.remote())

    # Wait for all workers to finish
    ray.get(futures)
