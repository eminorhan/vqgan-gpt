#!/bin/bash

#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=200GB
#SBATCH --time=0:30:00
#SBATCH --array=2-4
#SBATCH --job-name=sample_gpt
#SBATCH --output=sample_gpt_%A_%a.out

module purge
module load cuda/11.6.2

python -u /scratch/eo41/vqgan-gpt/sample.py \
	--condition 'free' \
	--n_samples 25 \
	--seed $SLURM_ARRAY_TASK_ID \
	--gpt_config 'GPT_gimel' \
	--gpt_model 'say_gimel' \
	--data_path ''

echo "Done"
