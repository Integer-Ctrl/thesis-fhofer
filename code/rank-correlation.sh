#!/bin/bash
#SBATCH --job-name=rank_correlation                   # Job name
#SBATCH --partition=gammaweb                          # Partition name
#SBATCH --exclude=gammaweb10                          # Exclude gammaweb10 node
#SBATCH --mem=16G                                     # Memory request (128GB)
#SBATCH --array=1-11                                  # Array job with 100 tasks
#SBATCH --ntasks=1                                    # Number of tasks (1 job/task)
#SBATCH --nodes=1                                     # Number of nodes
#SBATCH --cpus-per-task=2                             # Number of CPU cores per task
#SBATCH --time=48:00:00                               # Time limit (48 hours)
#SBATCH --output=logs/rank_correlation/%A_%a.out      # Standard output log
#SBATCH --error=logs/rank_correlation/%A_%a.err       # Standard error log

echo ${SLURM_ARRAY_JOB_ID}

# Load Python virtual environment
echo "Loading Python virtual environment..."
source ../thesis-fhofer/pyenv/bin/activate
echo "Python virtual environment loaded."

# Step 1: Compute rank correlation for passage-to-document conversion
echo "Running rank_correlation_pq.py..."
python3 -u src/passages_to_document/rank_correlation_pq.py ${SLURM_ARRAY_TASK_ID} ${SLURM_ARRAY_TASK_MAX}
echo "Rank correlation computed."

# Save logs, only first job responsible for creating the folder
if [ "${SLURM_ARRAY_TASK_ID}" -eq 1 ]; then
    mkdir -p logs/rank_correlation/${SLURM_ARRAY_JOB_ID}
fi

# Move logs to the folder if it exists
if [ -d "logs/rank_correlation/${SLURM_ARRAY_JOB_ID}" ]; then
    mv logs/rank_correlation/${SLURM_ARRAY_JOB_ID}_*.out logs/rank_correlation/${SLURM_ARRAY_JOB_ID}
    mv logs/rank_correlation/${SLURM_ARRAY_JOB_ID}_*.err logs/rank_correlation/${SLURM_ARRAY_JOB_ID}
fi