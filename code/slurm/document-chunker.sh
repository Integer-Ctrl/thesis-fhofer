#!/bin/bash
#SBATCH --job-name=chunker                              # Job name
#SBATCH --partition=gammaweb                            # Partition name
#SBATCH --exclude=gammaweb10                            # Exclude gammaweb10 node
#SBATCH --mem=64G                                       # Memory request (64GB)
#SBATCH --ntasks=1                                      # Number of tasks (1 job/task)
#SBATCH --nodes=1                                       # Number of nodes
#SBATCH --time=48:00:00                                 # Time limit (48 hours)
#SBATCH --output=../logs/document_chunker/slurm_%j.out  # Standard output log
#SBATCH --error=../logs/document_chunker/slurm_%j.err   # Standard error log

# Load Python virtual environment
echo "Loading Python virtual environment..."
source ../../../thesis-fhofer/pyenv/bin/activate
echo "Python virtual environment loaded."

# Step 1: Initialize folder structure
echo "Running init_folder_structure.py..."
python3 -u ../src/init_folder_structure.py
echo "Folder structure initialized."

# Step 2: Run document to passages chunker
echo "Running document_chunker.py..."
python3 -u ../src/document_chunker/document_chunker.py
echo "Document to passages conversion completed."

# Step 3: Build index
echo "Building index..."
python3 -u ../src/build_index.py
echo "Index built."