# thesis-fhofer

## Name
Transferring Relevance Judgments with Pairwise Preferences

## Description
This repository contains the materials for my bachelor thesis, which explores the transferability of relevance judgements from one dataset to another. The work is divided into two parts:

1. **Thesis Document**  
   The full written thesis, detailing the theoretical background, methodology, experiments, and findings.

2. **Source Code**  
   The implementation of the practical experiments conducted as part of the thesis. The source code includes scripts for transferring relevance judgements, evaluating the quality of the inferred judgements on the target dataset.

## Installation

Install all required packages with `pip install -r code/requirements.txt`.

## Usage

### Configuration

Before running any scirpts please configure your project.

**General:**
- DATA_PATH

**Source dataset:**
- DOCUMENT_DATASET_SOURCE_NAME
- DOCUMENT_DATASET_SOURCE_NAME_PYTERRIER
- DOCUMENT_DATASET_SOURCE_NAME_PYTHON_API

**Target datset** (can be the same as source dataset)
- DOCUMENT_DATASET_TARGET_NAME
- DOCUMENT_DATASET_TARGET_NAME_PYTERRIER
- DOCUMENT_DATASET_TARGET_NAME_PYTHON_API

If you want to use chatnoir for the target dataset pleases provide following configuration:

**ChatNoir** (optional)
- CHATNOIR_RETRIEVAL
- CHATNOIR_TARGET_INDICES
- CHATNOIR_API_KEY

### Dataset Segmentation (serial)
To preprocess the documents of the source dataset run the following scripts:
1. `code/src/init_folderstructure.py`
2. `code/src/passage_chunker/passage_chunker_serial.py`. \
3. `code/src/build_index/build_index.py`
If you are using slurm you can just submit `sbatch code/src/passage-chunker.sh` to your job manager.
- **WARNING**: Configure the loading of the _venv_ before submitting `passage-chunker.sh`:
   ```
   # Load Python virtual environment
   echo "Loading Python virtual environment..."
   source path/to/pyenv/bin/activate
   echo "Python virtual environment loaded."
   ```

### Passage Scoring (parallel)
To assign scores to the segmented passages run `code/src/passage_scorer/passage_scorer.py JOB_ID NUM_JOBS`
If you are using slurm you can just submit `sbatch code/src/passage-scorer.sh` to your job manager.
- **WARNING**: Configure the loading of the _venv_ before submitting `passage-scorer.sh`:
   ```
   # Load Python virtual environment
   echo "Loading Python virtual environment..."
   source path/to/pyenv/bin/activate
   echo "Python virtual environment loaded."
   ```

### Evaluation of Passage Scoring (parallel/serial)
In order to evaluate the quality of the assigned passage scores run the following scripts: 
1. `code/src/passage_scorer/evaluation/rank_correlation_pq.py JOB_ID NUM_JOBS` (parallel)
2. `code/src/passage_scorer/evaluation/cross_validation.py` (serial)
3. `code/src/passage_scorer/evaluation/cross_validation_scores.py` (serial)

If you are using slurm you can just submit the following jobs to your job manager: 
1. `sbatch code/src/cross-validation-scores.sh`
2. `sbatch code/src/rank-correlation-scores.sh`
- **WARNING**: Configure the loading of the _venv_ before submitting `cross-validation-scores.sh` and `rank-correlation-scores.sh`:
   ```
   # Load Python virtual environment
   echo "Loading Python virtual environment..."
   source path/to/pyenv/bin/activate
   echo "Python virtual environment loaded."
   ```

### Candidate Retrieval (serial)
To select candidates for each query of the retrieval task from the source dataset, run `code/src/pairwise_preference/candidate_retrieval.py`.
If you are using slurm you can just submit `sbatch code/src/candidate-retrieval.sh` to your job manager.
- **WARNING**: Configure the loading of the _venv_ before submitting `candidate-retrieval.sh`:
   ```
   # Load Python virtual environment
   echo "Loading Python virtual environment..."
   source path/to/pyenv/bin/activate
   echo "Python virtual environment loaded."
   ```

### Pairwise Preferences
tbd

### Evaluation of Pairwise Preferences
tbd

## Authors and acknowledgment
Fabian Hofer

## License
tbd

## Project status
tbd
