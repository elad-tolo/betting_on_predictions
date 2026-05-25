#!/bin/bash
#SBATCH --cpus-per-task=4        # Request 4 CPU cores
#SBATCH --gres=gpu:1             # Request 1 GPU
#SBATCH --job-name=multivariate_hc
#SBATCH --output=logs/%j_%x.log   # Save output in logs directory with job ID
#SBATCH --error=logs/%j_%x.log    # Redirect stderr to the same file
#SBATCH --ntasks=1

# Load Conda environment (adjust based on your environment needs)
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate ml_hypothesis

echo "Running multivariate_health_care.py with arguments: $@"
python -u multivariate_health_care.py "$@"
echo "Finished running multivariate_health_care.py"
