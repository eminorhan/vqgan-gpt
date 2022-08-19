#!/bin/bash

#SBATCH --account=cds
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=200GB
#SBATCH --time=1:00:00
#SBATCH --array=0
#SBATCH --job-name=test_encoder
#SBATCH --output=test_encoder_%A_%a.out

module purge
module load cuda/11.3.1

python -u /scratch/eo41/vqgan-gpt/test_encoder.py 

echo "Done"
