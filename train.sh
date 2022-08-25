#!/bin/bash

##SBATCH --account=cds
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=492GB
#SBATCH --time=48:00:00
#SBATCH --array=0
#SBATCH --job-name=train_gpt
#SBATCH --output=train_gpt_%A_%a.out

### change WORLD_SIZE as gpus/node * num_nodes
export MASTER_ADDR=$(hostname -s)
export MASTER_PORT=$(shuf -i 10000-65500 -n 1)
export WORLD_SIZE=16

module purge
module load cuda/11.3.1

LR=0.0005
OPTIMIZER='Adam'

srun python -u /scratch/eo41/vqgan-gpt/train.py --save_dir '/scratch/eo41/vqgan-gpt/gpt_pretrained_models' --batch_size 4 --optimizer $OPTIMIZER --lr $LR --seed $SLURM_ARRAY_TASK_ID --n_layer 48 --n_head 16 --n_emb 1600 --num_workers 8 --resume ""

echo "Done"
