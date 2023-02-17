#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=250GB
#SBATCH --time=1:00:00
#SBATCH --array=0
#SBATCH --job-name=finetune_gpt
#SBATCH --output=finetune_gpt_%A_%a.out

### change WORLD_SIZE as gpus/node * num_nodes
export MASTER_ADDR=$(hostname -s)
export MASTER_PORT=$(shuf -i 10000-65500 -n 1)
export WORLD_SIZE=2

module purge
module load cuda/11.6.2

LR=0.0003
OPTIMIZER='Adam'

# ####################### SAY #######################
srun python -u /scratch/eo41/vqgan-gpt/finetune.py \
	--save_dir '/scratch/eo41/vqgan-gpt/gpt_finetuned_models' \
	--batch_size 6 \
	--gpt_config 'GPT_gimel' \
	--num_workers 16 \
	--optimizer ${OPTIMIZER} \
	--lr ${LR} \
	--seed ${SLURM_ARRAY_TASK_ID} \
	--data_path '/vast/eo41/data/konkle_split/train' \
	--vqconfig_path '/scratch/eo41/vqgan-gpt/vqgan_pretrained_models/say_32x32_8192.yaml' \
	--vqmodel_path '/scratch/eo41/vqgan-gpt/vqgan_pretrained_models/say_32x32_8192.ckpt' \
	--resume '/scratch/eo41/vqgan-gpt/gpt_pretrained_models/say_gimel.pt' \
	--save_prefix 'say_gimel_konkle'

# # ####################### SFP #######################
# srun python -u /scratch/eo41/vqgan-gpt/finetune.py \
# 	--save_dir '/scratch/eo41/vqgan-gpt/gpt_finetuned_models' \
# 	--batch_size 6 \
# 	--gpt_config 'GPT_gimel' \
# 	--num_workers 16 \
# 	--optimizer ${OPTIMIZER} \
# 	--lr ${LR} \
# 	--seed ${SLURM_ARRAY_TASK_ID} \
#   --data_path '/vast/eo41/data/konkle_split/train' \
# 	--vqconfig_path '/scratch/eo41/vqgan-gpt/vqgan_pretrained_models/s_32x32_8192.yaml' \
# 	--vqmodel_path '/scratch/eo41/vqgan-gpt/vqgan_pretrained_models/s_32x32_8192.ckpt' \
# 	--resume '/scratch/eo41/vqgan-gpt/gpt_pretrained_models/s_gimel.pt' \
# 	--save_prefix 's_gimel_konkle'

# # # ####################### A #######################
# srun python -u /scratch/eo41/vqgan-gpt/finetune.py \
# 	--save_dir '/scratch/eo41/vqgan-gpt/gpt_finetuned_models' \
# 	--batch_size 6 \
# 	--gpt_config 'GPT_gimel' \
# 	--num_workers 16 \
# 	--optimizer ${OPTIMIZER} \
# 	--lr ${LR} \
# 	--seed ${SLURM_ARRAY_TASK_ID} \
#   --data_path '/vast/eo41/data/konkle_split/train' \
# 	--vqconfig_path '/scratch/eo41/vqgan-gpt/vqgan_pretrained_models/a_32x32_8192.yaml' \
# 	--vqmodel_path '/scratch/eo41/vqgan-gpt/vqgan_pretrained_models/a_32x32_8192.ckpt' \
# 	--resume '/scratch/eo41/vqgan-gpt/gpt_pretrained_models/a_gimel.pt' \
#  	--save_prefix 'a_gimel_konkle'

# # ####################### Y #######################
# srun python -u /scratch/eo41/vqgan-gpt/finetune.py \
# 	--save_dir '/scratch/eo41/vqgan-gpt/gpt_finetuned_models' \
# 	--batch_size 6 \
# 	--gpt_config 'GPT_gimel' \
# 	--num_workers 16 \
# 	--optimizer ${OPTIMIZER} \
# 	--lr ${LR} \
# 	--seed ${SLURM_ARRAY_TASK_ID} \
#   --data_path '/vast/eo41/data/konkle_split/train' \
# 	--vqconfig_path '/scratch/eo41/vqgan-gpt/vqgan_pretrained_models/y_32x32_8192.yaml' \
# 	--vqmodel_path '/scratch/eo41/vqgan-gpt/vqgan_pretrained_models/y_32x32_8192.ckpt' \
# 	--resume '/scratch/eo41/vqgan-gpt/gpt_pretrained_models/y_gimel.pt' \
# 	--save_prefix 'y_gimel_konkle'

echo "Done"