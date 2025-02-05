#!/bin/bash
#SBATCH --job-name=monoprompt                               # Job name
#SBATCH --partition=gammaweb                                # Partition name
#SBATCH --gres=gpu:3g.20gb:1                                # Request GPU resource
#SBATCH --mem=128                                           # Memory request (128GB)
#SBATCH --ntasks=1                                          # Number of tasks (1 job/task)
#SBATCH --nodes=1                                           # Number of nodes
#SBATCH --cpus-per-task=2                                   # Number of CPU cores per task
#SBATCH --time=48:00:00                                     # Time limit (48 hours)
#SBATCH --output=logs/pointwise_preference/slurm_%j.out     # Standard output log
#SBATCH --error=logs/pointwise_preference/slurm_%j.err      # Standard error log

# gpu:hopper:1
# gpu:3g.20gb:1 

# Load Python virtual environment
echo "Loading Python virtual environment..."
source ../../thesis-fhofer/pyenv/bin/activate
echo "Python virtual environment loaded."

# Run Pointwise preference with monoprompt on candidates
echo "Running pointwise_preference_monoprompt.py..."
python3 -u src/pointwise_preference/pointwise_preference_monoprompt.py
echo "Pointwise preferences completed."