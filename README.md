# thesis-fhofer

## Description
This repository contains the materials for my bachelor thesis, which explores the transferability of relevance judgments from one dataset to another. The work is divided into two parts:

### 1. Thesis Document
   The complete written thesis, detailing the theoretical foundations, methodology, experiments, and findings.

   **Title**\
   Transferring Relevance Judgments with Pairwise Preferences

   **Abstract**\
   Relevance judgments are essential for evaluating and comparing the effectiveness of information retrieval systems. Traditionally, human assessors manually review query-document pairs to determine relevance judgments. This process is costly, time-consuming, and must be repeated for each new test collection. This thesis addresses this challenge by presenting a method for automatically generating relevance judgments using existing annotated datasets. The proposed approach transfers relevance information from a well-judged source corpus to a subset of documents in a target corpus, enriching it with newly generated judgments. The method employs a pairwise preference approach, where a large language model compares already judged documents from the source corpus with candidate documents from the target corpus. To do this, the model is prompted to determine whether a target document is as relevant to a given query as an already judged source document, resulting in an automatically generated set of relevance judgments for the target corpus. To evaluate the effectiveness of the developed approach, the transfer method is applied to multiple existing test collections that already contain relevance judgments. Each collection serves as both a source and its own target corpus, enabling automatic evaluation by comparing the newly generated judgments with the original ones. Additionally, the approach is tested on \texttt{ClueWeb22/b} as an unjudged target corpus. By leveraging pairwise preference with already judged documents, this approach has the potential to significantly reduce the effort for manual annotation while maintaining high-quality relevance judgments, making scalable enrichment of target corpora possible.

### 2. Source Code
   The implementation of the experiments conducted for the thesis, including scripts for transferring relevance judgments and evaluating the quality of inferred judgments on target datasets.


## Transfer Pipeline

An automated transfer pipeline was developed to move relevance judgments from an existing test collection to a target corpus using pairwise preferences. Before running the pipeline, the project must be properly configured.

### Configuration

Please ensure the following arguments are set before executing any scripts:

#### Args
- DATA_PATH: Absolute path where datasets and pipeline outputs are stored. 
   > [!IMPORTANT]
   > This is an absolute path. All other configuration paths are located in this path.
   - e.g.: `/dev/ir/relevance-transfer/data`

- **CHATNOIR_RETRIEVAL**: Whether to use ChatNoir for *passage scoring* and *candidate retrieval*.  
  Example: `false`

- **CHATNOIR_SOURCE_INDICES**: Source dataset index for ChatNoir retrieval (only needed if `CHATNOIR_RETRIEVAL=true`).  
  Example: `clueweb22/b`

- **CHATNOIR_TARGET_INDICES**: Target corpus index for ChatNoir retrieval.  
  Example: `clueweb22/b`

- **QUERIES**: List of query IDs for which relevance judgments should be generated. Empty list `[]` processes all queries.  
  Example: `[23, 55, 97]`

- **CHATNOIR_API_KEY**: API key for ChatNoir (optional, for higher rate limits).  
  Example: `<your_api_key>`

- **PASSAGE_ID_SEPARATOR**: Separator for uniquely identifying passages.  
  Example: `___`

- **KEY_SEPARATOR**: Internal separator for data handling.  
  Example: `---`

- **NUMBER_OF_CROSS_VALIDATION_FOLDS**: Number of folds for cross-validation evaluation.  
  Example: `5`

- **PT_RETRIEVERS**: Retrieval models used from PyTerrier for passage scoring.  
  Example: `["BM25", "DFR_BM25", "DFIZ"]`

- **AGGREGATION_METHODS**: Methods for aggregating passage relevance scores.  
  Example: `["mean", "max", "min", "sum"]`

- **TRANSFORMATION_METHODS**: Methods to transform scores before evaluation.  
  Example: `["id", "log", "exp", "sqrt"]`

- **EVALUATION_METHODS**: Rank correlation methods used for evaluation.  
  Example: `["pearson", "spearman", "kendall"]`

- **METRICS**: Metrics used during passage scoring.  
  Example: `["p10", "p10_wod", "ndcg10", "ndcg10_wod"]`

**Source Dataset Settings**

- **DOCUMENT_DATASET_SOURCE_NAME**: ir_datasets name of the source dataset.  
  Example: `argsme/2020-04-01/touche-2020-task-1`

- **DOCUMENT_DATASET_SOURCE_NAME_PYTERRIER**: PyTerrier name for source dataset.  
  Example: `irds:argsme/2020-04-01/touche-2020-task-1`

- **DOCUMENT_DATASET_SOURCE_NAME_PYTHON_API**: Python API name for source dataset.  
  Example: `argsme/2020-04-01/touche-2020-task-1`

- **DOCUMENT_DATASET_SOURCE_INDEX_PATH**: Path to source dataset document index.  
  Example: `document-indices`

- **PASSAGE_DATASET_SOURCE_PATH**: Path to segmented source passages.  
  Example: `passages.jsonl.gz`

- **PASSAGE_DATASET_SOURCE_INDEX_PATH**: Path to source passage index.  
  Example: `passage-indices`

- **PASSAGE_DATASET_SOURCE_SCORE_REL_PATH**: Path to scores for relevant passages only.  
  Example: `retrieval-scores-rel/`

- **PASSAGE_DATASET_SOURCE_SCORE_AQ_PATH**: Path to scores for all retrieved passages.  
  Example: `retrieval-scores-aq/`

- **RANK_CORRELATION_SCORE_PQ_AQ_PATH**: Path for rank correlation scores per query.  
  Example: `rank-correlation-scores-pq-aq/`

- **RANK_CORRELATION_SCORE_AVG_PATH**: File for average rank correlation scores.  
  Example: `avg-rank-correlation-scores.jsonl.gz`

**Target Dataset Settings**

- **DOCUMENT_DATASET_TARGET_NAME**: ir_datasets name of the target dataset.  
  Example: `argsme/2020-04-01/touche-2020-task-1`

- **DOCUMENT_DATASET_TARGET_NAME_PYTERRIER**: PyTerrier name for the target dataset.  
  Example: `irds:argsme/2020-04-01/touche-2020-task-1`

- **DOCUMENT_DATASET_TARGET_NAME_PYTHON_API**: Python API name for the target dataset.  
  Example: `argsme/2020-04-01/touche-2020-task-1`

- **DOCUMENT_DATASET_TARGET_INDEX_PATH**: Path to target dataset document index.  
  Example: `document-indices`

- **PASSAGE_DATASET_TARGET_PATH**: Path to segmented target passages.  
  Example: `passages.jsonl.gz`

- **PASSAGE_DATASET_TARGET_INDEX_PATH**: Path to target passage index.  
  Example: `passage-indices`

- **LABEL_RANK_CORRELATION_SCORE_PQ_AQ_PATH**: Path for rank correlation scores of inferred judgments.  
  Example: `rank-correlation-scores-pq-aq/`

- **LABEL_CROSS_VALIDATION_PATH**: Path for cross-validation results of inferred scores.  
  Example: `cross-validation/`

- **LABEL_CORRELATION_AVG_PER_QUERY_PATH**: Path for average evaluation scores per query.  
  Example: `avg-per-query/`

- **CANDIDATES_LOCAL_PATH**: Path for candidates retrieved via local index.  
  Example: `candidates/`

- **CANDIDATE_CHATNOIR_PATH**: Path for candidates retrieved via ChatNoir.  
  Example: `candidates-chatnoir/`

**Inference and LLM Settings**

- **PREFERENCE_BACKBONE**: Backbone LLM for pairwise preference inference.  
  Example: `google/flan-t5-base`

- **BACKBONES**: List of LLM models used for evaluation.  
  Example: `["google/flan-t5-base", "google/flan-t5-small", "google-t5/t5-small"]`

- **ONLY_JUDGED**: Toggle to only infer relevance scores for already judged documents.  
  Example: `true`

- **DUOPROMPT_PATH**: Directory for DuoPrompt inferred relevance scores.  
  Example: `duoprompt/`

- **DUOPROMPT_CACHE_NAME**: Cache file for DuoPrompt inferences.  
  Example: `duoprompt.jsonl.cache.gz`

- **MONOPROMPT_PATH**: Directory for MonoPrompt inferred relevance scores.  
  Example: `monoprompt/`

- **MONOPROMPT_CACHE_NAME**: Cache file for MonoPrompt inferences.  
  Example: `monoprompt.jsonl.cache.gz`


#### Example config

```json
"DATA_PATH": "",

"CHATNOIR_RETRIEVAL": false,
"CHATNOIR_SOURCE_INDICES": "",
"CHATNOIR_TARGET_INDICES": "clueweb22/b",
"QUERIES": [],
"CHATNOIR_API_KEY": "",

"PASSAGE_ID_SEPARATOR": "___",
"KEY_SEPARATOR": "---",

"NUMBER_OF_CROSS_VALIDATION_FOLDS": 5,

"PT_RETRIEVERS": ["BM25", "DFR_BM25", "DFIZ", "DLH", "DPH", "DirichletLM", "Hiemstra_LM", "LGD", "PL2", "TF_IDF"],
"AGGREGATION_METHODS": ["mean", "max", "min", "sum"],
"TRANSFORMATION_METHODS": ["id", "log", "exp", "sqrt"],
"EVALUATION_METHODS": ["pearson", "spearman", "kendall", "pearson-greedy", "spearman-greedy", "kendall-greedy"],
"METRICS": ["p10", "p10_wod", "ndcg10", "ndcg10_wod", "reciprocal_rank_docno"],



"TYPE_SOURCE": "document",
"DOCUMENT_DATASET_SOURCE_NAME": "",
"DOCUMENT_DATASET_SOURCE_NAME_PYTERRIER": "irds:",
"DOCUMENT_DATASET_SOURCE_NAME_PYTHON_API": "",
"DOCUMENT_DATASET_SOURCE_INDEX_PATH": "document-indices",

"PASSAGE_DATASET_SOURCE_PATH": "passages.jsonl.gz",
"PASSAGE_DATASET_SOURCE_INDEX_PATH": "passage-indices",
"PASSAGE_DATASET_SOURCE_SCORE_REL_PATH": "retrieval-scores-rel/",
"PASSAGE_DATASET_SOURCE_SCORE_AQ_PATH": "retrieval-scores-aq/",

"RANK_CORRELATION_SCORE_PQ_AQ_PATH": "rank-correlation-scores-pq-aq/",
"RANK_CORRELATION_SCORE_AVG_PATH": "avg-rank-correlation-scores.jsonl.gz",



"TYPE_TARGET": "document",
"DOCUMENT_DATASET_TARGET_NAME": "",
"DOCUMENT_DATASET_TARGET_NAME_PYTERRIER": "irds:",
"DOCUMENT_DATASET_TARGET_NAME_PYTHON_API": "",
"DOCUMENT_DATASET_TARGET_INDEX_PATH": "document-indices",

"PASSAGE_DATASET_TARGET_PATH": "passages.jsonl.gz",
"PASSAGE_DATASET_TARGET_INDEX_PATH": "passage-indices",

"LABEL_RANK_CORRELATION_SCORE_PQ_AQ_PATH": "rank-correlation-scores-pq-aq/",
"LABEL_CROSS_VALIDATION_PATH": "cross-validation/",
"LABEL_CORRELATION_AVG_PER_QUERY_PATH": "avg-per-query/",

"CANDIDATES_LOCAL_PATH": "candidates/",
"CANDIDATE_CHATNOIR_PATH": "candidates-chatnoir/",

"PREFERENCE_BACKBONE": "google/flan-t5-small",
"BACKBONES": ["google/flan-t5-base", "google/flan-t5-small", "google-t5/t5-small"],
"ONLY_JUDGED": true,

"DUOPROMPT_PATH": "duoprompt/",
"DUOPROMPT_CACHE_NAME": "duoprompt.jsonl.cache.gz",

"MONOPROMPT_PATH": "monoprompt/",
"MONOPROMPT_CACHE_NAME": "monoprompt.jsonl.cache.gz"
```

## Pipeline Stages

1. Dataset Selection and Segmentation
2. Passage Scoring
   - Evaluation of Passage Scoring
3. Candidate Retrieval
4. Pairwise Preferences
   - Evaluation of Pairwise Preferences

## Project status
Finished. Submission date 13/03/2025
