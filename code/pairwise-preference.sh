#!/bin/bash
#SBATCH --job-name=duoprompt                      # Job name
#SBATCH --partition=gammaweb                                # Partition name
#SBATCH --gres=gpu:hopper:1                                 # Request GPU resource
#SBATCH --mem=128G                                          # Memory request (128GB)
#SBATCH --ntasks=1                                          # Number of tasks (1 job/task)
#SBATCH --nodes=1                                           # Number of nodes
#SBATCH --cpus-per-task=2                                   # Number of CPU cores per task
#SBATCH --time=48:00:00                                     # Time limit (48 hours)
#SBATCH --output=logs/pairwise_preference/slurm_%j.out      # Standard output log
#SBATCH --error=logs/pairwise_preference/slurm_%j.err       # Standard error log

# gpu:3g.20gb:1 

# Load Python virtual environment
echo "Loading Python virtual environment..."
source ../../thesis-fhofer/pyenv/bin/activate
echo "Python virtual environment loaded."

# Run pairwise preference with duoprompt on candidates
echo "Running pairwise_preference_duoprompt.py..."
python3 -u src/pairwise_preference/pairwise_preference_duoprompt.py
echo "Pairwise preferences completed."