#!/bin/bash
#SBATCH --job-name=passage_pipeline             # Job name
#SBATCH --partition=gammaweb                    # Partition name
#SBATCH --exclude=gammaweb10                    # Exclude gammaweb10 node
#SBATCH --array=1-51                            # Array job with 100 tasks
#SBATCH --mem=12G                               # Memory request
#SBATCH --ntasks=1                              # Number of tasks
#SBATCH --nodes=1                               # Number of nodes
#SBATCH --cpus-per-task=2                       # Number of CPU cores per task
#SBATCH --time=48:00:00                         # Time limit
#SBATCH --output=logs/%A_%a.out                 # Logs will be stored in this format
#SBATCH --error=logs/%A_%a.err                  # Error logs will be stored here

echo ${SLURM_ARRAY_JOB_ID}

# Load Python virtual environment
echo "Loading Python virtual environment..."
source ../thesis-fhofer/pyenv/bin/activate
echo "Python virtual environment loaded."

# Step 1: Initialize folder structure
echo "Running init_folder_structure.py..."
python3 -u code/src/init_folder_structure.py
echo "Folder structure initialized."

# Step 2: Run passage scorer with the task ID as an argument
echo "Running passage_scorer.py..."
python3 -u code/src/passage_scorer/passage_scorer.py ${SLURM_ARRAY_TASK_ID} ${$SLURM_ARRAY_TASK_MAX}
echo "Passage scoring completed."

# Save logs, only first job responsible for creating the folder
if [ "${SLURM_ARRAY_TASK_ID}" -eq 1 ]; then
    mkdir -p logs/${SLURM_ARRAY_JOB_ID}
fi

# Move logs to the folder if it exists
if [ -d "logs/${SLURM_ARRAY_JOB_ID}" ]; then
    mv logs/${SLURM_ARRAY_JOB_ID}_*.out logs/${SLURM_ARRAY_JOB_ID}
    mv logs/${SLURM_ARRAY_JOB_ID}_*.err logs/${SLURM_ARRAY_JOB_ID}
fi
