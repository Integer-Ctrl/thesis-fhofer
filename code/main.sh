#!/bin/bash
#SBATCH --job-name=main_job                  # Name of the main job
#SBATCH --output=main_job_%j.out             # Output log for the main job
#SBATCH --error=main_job_%j.err              # Error log for the main job
#SBATCH --partition=gammaweb                 # Partition name
#SBATCH --exclude=gammaweb10                 # Exclude gammaweb10 node
#SBATCH --time=7-00:00                       # Time limit
#SBATCH --mem=2G                             # Memory request
#SBATCH --ntasks=1                           # Number of tasks

echo "Starting main job at $(date)"

# Step 1: Run passage chunker
job1_id=$(sbatch passage-chunker.sh)
echo "Passage chunker job submitted with ID: ${job1_id}"

# Step 2: Run passage scorer
job2_id=$(sbatch --dependency=afterok:${job1_id} passage-scorer.sh)
echo "Passage scorer job submitted with ID: ${job2_id}"

# Step 3: Run rank correlation per query
job3_id=$(sbatch --dependency=afterok:${job2_id} rank-correlation.sh)
echo "Rank correlation job submitted with ID: ${job3_id}"

# Step 4: Run cross-validation
job4_id=$(sbatch --dependency=afterok:${job3_id} cross-validation.sh)
echo "Cross-validation job submitted with ID: ${job4_id}"

# Step 5: Run candidate retrieval
job5_id=$(sbatch --dependency=afterok:${job4_id} candidate-retrieval.sh)
echo "Candidate retrieval job submitted with ID: ${job5_id}"

echo "Main job completed at $(date)"

