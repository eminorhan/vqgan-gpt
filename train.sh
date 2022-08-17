#!/bin/bash

#SBATCH --account=cds
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:rtx8000:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=300GB
#SBATCH --time=02:00:00
#SBATCH --array=0
#SBATCH --job-name=train_gpt
#SBATCH --output=train_gpt_%A_%a.out

### change WORLD_SIZE as gpus/node * num_nodes
export MASTER_ADDR=$(hostname -s)
export MASTER_PORT=$(shuf -i 10000-65500 -n 1)
export WORLD_SIZE=2

module purge
module load cuda/11.3.1

LR=0.0005
OPTIMIZER='Adam'

srun python -u /scratch/eo41/vqgan-gpt/train.py --save_dir '/scratch/eo41/vqgan-gpt/gpt_pretrained_models' --batch_size 2 --optimizer $OPTIMIZER --lr $LR --seed $SLURM_ARRAY_TASK_ID --n_layer 48 --n_head 16 --n_emb 1024 --resume ''

echo "Done"
