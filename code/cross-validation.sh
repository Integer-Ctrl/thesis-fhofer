#!/bin/bash
#SBATCH --job-name=cross_validation                   # Job name
#SBATCH --partition=gammaweb                          # Partition name
#SBATCH --exclude=gammaweb10                          # Exclude gammaweb10 node
#SBATCH --mem=128G                                     # Memory request (128GB)
#SBATCH --ntasks=1                                    # Number of tasks (1 job/task)
#SBATCH --nodes=1                                     # Number of nodes
#SBATCH --cpus-per-task=2                             # Number of CPU cores per task
#SBATCH --time=48:00:00                               # Time limit (48 hours)
#SBATCH --output=logs/cross_validation/slurm_%j.out   # Standard output log
#SBATCH --error=logs/cross_validation/slurm_%j.err       # Standard error log

echo ${SLURM_ARRAY_JOB_ID}

# Load Python virtual environment
echo "Loading Python virtual environment..."
source ../../thesis-fhofer/pyenv/bin/activate
echo "Python virtual environment loaded."

# Step 1: Run cross-validation for evaluation
echo "Running cross_validation.py..."
python3 -u src/passages_to_document/cross_validation.py
echo "Cross-validation completed."

# Step 2: Run cross-validation-scores for evaluation
echo "Running cross_validation_scores.py..."
python3 -u src/passages_to_document/cross_validation_scores.py
echo "Cross-validation scores computed."