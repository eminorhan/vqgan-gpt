#!/bin/bash

#SBATCH --account=cds
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=200GB
#SBATCH --time=01:00:00
#SBATCH --array=0
#SBATCH --job-name=test_shuffle
#SBATCH --output=test_shuffle_%A_%a.out

module purge
module load cuda/11.3.1

python -u /scratch/eo41/vqgan-gpt/tests/test_shuffle.py --data_path "/scratch/eo41/data/imagenet/imagenet_val_000000.tar" --n_samples 9999

echo "Done"
