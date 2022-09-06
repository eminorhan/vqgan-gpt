#!/bin/bash

##SBATCH --account=cds
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=200GB
#SBATCH --time=00:10:00
#SBATCH --array=0
#SBATCH --job-name=train_gpt
#SBATCH --output=train_gpt_%A_%a.out

### change WORLD_SIZE as gpus/node * num_nodes
export MASTER_ADDR=$(hostname -s)
export MASTER_PORT=$(shuf -i 10000-65500 -n 1)
export WORLD_SIZE=2

module purge
module load cuda/11.3.1

LR=0.0003
OPTIMIZER='Adam'

srun python -u /scratch/eo41/vqgan-gpt/train.py \
	--save_dir '/scratch/eo41/vqgan-gpt/gpt_pretrained_models' \
	--batch_size 64 \
	--n_layer 12 \
	--n_head 12 \
	--n_emb 768 \
	--num_workers 8 \
	--optimizer $OPTIMIZER \
	--lr $LR \
	--seed $SLURM_ARRAY_TASK_ID \
	--resume ''

echo "Done"
