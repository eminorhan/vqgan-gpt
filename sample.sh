#!/bin/bash

##SBATCH --account=cds
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=200GB
#SBATCH --time=0:10:00
#SBATCH --array=0
#SBATCH --job-name=sample_igpt
#SBATCH --output=sample_igpt_%A_%a.out

module purge
module load cuda/11.3.1

python -u /scratch/eo41/vqgan-gpt/sample.py

echo "Done"
