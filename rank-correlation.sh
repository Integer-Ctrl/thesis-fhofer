#!/bin/bash
#SBATCH --job-name=rank_correlation   # Job name
#SBATCH --partition=gammaweb          # Partition name
#SBATCH --exclude=gammaweb10          # Exclude gammaweb10 node
#SBATCH --mem=128G                    # Memory request (128GB)
#SBATCH --ntasks=1                    # Number of tasks (1 job/task)
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --cpus-per-task=32            # Number of CPU cores per task
#SBATCH --time=48:00:00               # Time limit (48 hours)
#SBATCH --output=logs/rank_correlation/slurm_%j.out         # Standard output log
#SBATCH --error=logs/rank_correlation/slurm_%j.err          # Standard error log

# Load Python virtual environment
echo "Loading Python virtual environment..."
source ../thesis-fhofer/pyenv/bin/activate
echo "Python virtual environment loaded."

# Step 1: Compute rank correlation for passage-to-document conversion
echo "Running rank_correlation_pq.py..."
python3 -u code/src/passages_to_document/rank_correlation_pq.py
echo "Rank correlation computed."

# Step 2: Run cross-validation for evaluation
echo "Running cross_validation.py..."
python3 -u code/src/passages_to_document/cross_validation.py
echo "Cross-validation completed."