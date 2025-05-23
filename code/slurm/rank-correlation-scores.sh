#!/bin/bash
#SBATCH --job-name=correlation                        # Job name
#SBATCH --partition=gammaweb                          # Partition name
#SBATCH --exclude=gammaweb10                          # Exclude gammaweb10 node
#SBATCH --mem=32G                                     # Memory request (128GB)
#SBATCH --ntasks=1                                    # Number of tasks (1 job/task)
#SBATCH --nodes=1                                     # Number of nodes
#SBATCH --cpus-per-task=2                             # Number of CPU cores per task
#SBATCH --time=48:00:00                               # Time limit (48 hours)
#SBATCH --output=slurm_%j.out                         # Standard output log
#SBATCH --error=slurm_%j.err                          # Standard error log


# Load Python virtual environment
echo "Loading Python virtual environment..."
source ../../../thesis-fhofer/pyenv/bin/activate
echo "Python virtual environment loaded."

# Step 1: Compute rank correlation for passage-to-document conversion
echo "Running rank_correlation_pq.py..."
python3 -u ../src/passage_scorer/evaluation/rank_correlation_pq.py
echo "Rank correlation computed."