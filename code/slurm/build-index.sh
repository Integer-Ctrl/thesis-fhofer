#!/bin/bash
#SBATCH --job-name=passage_chunker                   # Job name
#SBATCH --partition=gammaweb                         # Partition name
#SBATCH --exclude=gammaweb10                         # Exclude gammaweb10 node
#SBATCH --mem=128G                                   # Memory request (128GB)
#SBATCH --ntasks=1                                   # Number of tasks (1 job/task)
#SBATCH --nodes=1                                    # Number of nodes
#SBATCH --cpus-per-task=32                           # Number of CPU cores per task
#SBATCH --time=48:00:00                              # Time limit (48 hours)
#SBATCH --output=logs/passage_chunker/slurm_%j.out   # Standard output log
#SBATCH --error=logs/passage_chunker/slurm_%j.err    # Standard error log

# Step 3: Build index
echo "Building index..."
python3 -u src/build_index/build_index.py
echo "Index built."