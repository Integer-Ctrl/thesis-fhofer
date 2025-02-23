#!/bin/bash
#SBATCH --job-name=candidate                                # Job name
#SBATCH --partition=gammaweb                                # Partition name
#SBATCH --exclude=gammaweb10                                # Exclude gammaweb10 node
#SBATCH --mem=96G                                           # Memory request (48GB)
#SBATCH --ntasks=1                                          # Number of tasks (1 job/task)
#SBATCH --nodes=1                                           # Number of nodes
#SBATCH --cpus-per-task=2                                   # Number of CPU cores per task
#SBATCH --time=48:00:00                                     # Time limit (48 hours)
#SBATCH --output=../logs/candidate_retrieval/slurm_%j.out      # Standard output log
#SBATCH --error=../logs/candidate_retrieval/slurm_%j.err       # Standard error log

# Load Python virtual environment
echo "Loading Python virtual environment..."
source ../../../thesis-fhofer/pyenv/bin/activate
echo "Python virtual environment loaded."

# Step 1: Run canidate-retrieval for pairwise candidates
echo "Running candidate_retrieval.py..."
python3 -u ../src/candidate_retrieval/candidate_retrieval.py # Run for clueweb22
echo "Candidate retrieval completed."