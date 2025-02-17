#!/bin/bash
#SBATCH --job-name=main_job                  # Name of the main job
#SBATCH --output=main_job_%j.out             # Output log for the main job
#SBATCH --error=main_job_%j.err              # Error log for the main job
#SBATCH --partition=gammaweb                 # Partition name
#SBATCH --exclude=gammaweb10                 # Exclude gammaweb10 node
#SBATCH --time=48:00:00                       # Time limit
#SBATCH --mem=2G                             # Memory request
#SBATCH --ntasks=1                           # Number of tasks

echo "Starting main job at $(date)"

# Step 1: Run passage chunker
job1_id=$(sbatch passage-chunker.sh | awk '{print $4}')
echo "Passage chunker job submitted with ID: ${job1_id}"

# Step 2: Run passage scorer
job2_id=$(sbatch --dependency=afterok:${job1_id} passage-scorer.sh | awk '{print $4}')
echo "Passage scorer job submitted with ID: ${job2_id}"

# Step 3: Run rank correlation (scores) per query
job3_id=$(sbatch --dependency=afterok:${job2_id} rank-correlation-scores.sh | awk '{print $4}')
echo "Rank correlation (scores) job submitted with ID: ${job3_id}"

# Step 4: Run cross-validation-scores
job4_id=$(sbatch --dependency=afterok:${job3_id} cross-validation-scores.sh | awk '{print $4}')
echo "Cross-validation job submitted with ID: ${job4_id}"

# Step 5: Run candidate retrieval
job5_id=$(sbatch --dependency=afterok:${job4_id} candidate-retrieval.sh | awk '{print $4}')
echo "Candidate retrieval job submitted with ID: ${job5_id}"

# Step 6: Run pairwise preference
job6_id=$(sbatch --dependency=afterok:${job5_id} pairwise-preference.sh | awk '{print $4}')

# Step 7: Run rank correlation (duoprompt) per query
job7_id=$(sbatch --dependency=afterok:${job6_id} rank-correlation-duoprompt.sh | awk '{print $4}')

# Step 8: Run cross-validation-duoprompt
job8_id=$(sbatch --dependency=afterok:${job7_id} cross-validation-duoprompt.sh | awk '{print $4}')

# Step 9: Run pointwise preference
job9_id=$(sbatch --dependency=afterok:${job8_id} pointwise-preference.sh | awk '{print $4}')

# Step 10: Run rank correlation (monoprompt) per query
job10_id=$(sbatch --dependency=afterok:${job9_id} rank-correlation-monoprompt.sh | awk '{print $4}')

# Step 11: Run cross-validation-monoprompt
job11_id=$(sbatch --dependency=afterok:${job10_id} cross-validation-monoprompt.sh | awk '{print $4}')

echo "Main job completed at $(date)"

