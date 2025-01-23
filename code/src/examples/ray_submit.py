import ray
import subprocess
import sys

# Initialize Ray.
ray.init(ignore_reinit_error=True)


@ray.remote
def run_scorer_script(process_id, total_processes):
    """
    Function to run the passage_scorer.py script with the given process ID and total processes.
    """
    # Build the command to run scorer.py
    command = ['python3', 'ray_example.py', str(process_id), str(total_processes)]

    # Execute the script using subprocess
    result = subprocess.run(command, capture_output=True, text=True)

    # Check for errors and print output
    if result.returncode == 0:
        print(f"Process {process_id} completed successfully.")
        print(result.stdout)
    else:
        print(f"Process {process_id} failed with error:")
        print(result.stderr)


if __name__ == '__main__':
    # Get the total number of processes from the second argument
    total_processes = 50

    # Create Ray tasks for each process (from 1 to total_processes)
    futures = []
    for process_id in range(1, total_processes + 1):
        futures.append(run_scorer_script.remote(process_id, total_processes))

    # Wait for all tasks to complete and gather results
    ray.get(futures)
