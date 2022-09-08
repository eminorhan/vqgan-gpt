#!/bin/bash

##SBATCH --account=cds
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=200GB
#SBATCH --time=0:10:00
#SBATCH --array=0
#SBATCH --job-name=sample_gpt
#SBATCH --output=sample_gpt_%A_%a.out

module purge
module load cuda/11.3.1

python -u /scratch/eo41/vqgan-gpt/sample.py \
	--condition "cond" \
	--gpt_config "GPTBeta" \
	--gpt_model "model_150000_36l_12h_768e_192b_0.0005lr_Adamo_0s"

echo "Done"
