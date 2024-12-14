#!/bin/bash
#SBATCH --job-name=passage_pipeline   # Job name
#SBATCH --partition=gammaweb          # Partition name
#SBATCH --mem=512G                    # Memory request (128GB)
#SBATCH --ntasks=1                    # Number of tasks (1 job/task)
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --cpus-per-task=32            # Number of CPU cores per task
#SBATCH --time=48:00:00               # Time limit (48 hours)
#SBATCH --output=slurm_%j.out         # Standard output log
#SBATCH --error=slurm_%j.err          # Standard error log

# Load Python virtual environment
echo "Loading Python virtual environment..."
source ../thesis-fhofer/pyenv/bin/activate
echo "Python virtual environment loaded."

# Step 1: Initialize folder structure
echo "Running init_folder_structure.py..."
python3 -u code/src/init_folder_structure.py
echo "Folder structure initialized."

# Step 2: Run document to passages chunker
echo "Running document_chunker_parallel.py..."
python3 -u code/src/passage_chunker/document_chunker_parallel.py
echo "Document to passages conversion completed."

# Step 3: Run passage scorer
echo "Running passage_scorer_parallel.py..."
python3 -u code/src/passage_scorer/passage_scorer_parallel.py
echo "Passage scoring completed."

# # Step 4: Compute rank correlation for passage-to-document conversion
# echo "Running rank_correlation_pq.py..."
# python3 -u code/src/passages_to_document/rank_correlation_pq.py
# echo "Rank correlation computed."

# # Step 5: Run cross-validation for evaluation
# echo "Running cross_validation.py..."
# python3 -u code/src/passages_to_document/cross_validation
# echo "Cross-validation completed."

# # Step 6: Run canidate-retrieval for pairwise candidates
# echo "Running candidate_retrieval.py..."
# python3 -u code/src/pairwise_preference/candidate_retrieval.py
# echo "Candidate retrieval completed."