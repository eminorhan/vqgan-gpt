#!/bin/bash

#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=492GB
#SBATCH --time=48:00:00
#SBATCH --array=0
#SBATCH --job-name=train_gpt
#SBATCH --output=train_gpt_%A_%a.out

### change WORLD_SIZE as gpus/node * num_nodes
export MASTER_ADDR=$(hostname -s)
export MASTER_PORT=$(shuf -i 10000-65500 -n 1)
export WORLD_SIZE=16

LR=0.0003
OPTIMIZER='Adam'

# # ####################### SAY #######################
# srun python -u /scratch/eo41/vqgan-gpt/train.py \
# 	--save_dir '/scratch/eo41/vqgan-gpt/gpt_pretrained_models' \
# 	--batch_size 6 \
# 	--gpt_config 'GPT_gimel' \
# 	--num_workers 16 \
# 	--print_freq 15000 \
# 	--optimizer ${OPTIMIZER} \
# 	--lr ${LR} \
# 	--seed ${SLURM_ARRAY_TASK_ID} \
# 	--data_path '/scratch/eo41/data/saycam/SAY_5fps_300s_{000000..000009}.tar' \
# 	--vqconfig_path '/scratch/eo41/vqgan-gpt/vqgan_pretrained_models/say_32x32_8192.yaml' \
# 	--vqmodel_path '/scratch/eo41/vqgan-gpt/vqgan_pretrained_models/say_32x32_8192.ckpt' \
# 	--resume '/scratch/eo41/vqgan-gpt/gpt_pretrained_models/say_gimel.pt' \
# 	--save_prefix 'say_gimel'

# # ####################### SFP #######################
# srun python -u /scratch/eo41/vqgan-gpt/train.py \
# 	--save_dir '/scratch/eo41/vqgan-gpt/gpt_pretrained_models' \
# 	--batch_size 6 \
# 	--gpt_config 'GPT_gimel' \
# 	--num_workers 16 \
# 	--print_freq 15000 \
# 	--optimizer ${OPTIMIZER} \
# 	--lr ${LR} \
# 	--seed ${SLURM_ARRAY_TASK_ID} \
# 	--data_path '/scratch/eo41/data/saycam/Sfp_5fps_300s_{000000..000003}.tar' \
# 	--vqconfig_path '/scratch/eo41/vqgan-gpt/vqgan_pretrained_models/s_32x32_8192.yaml' \
# 	--vqmodel_path '/scratch/eo41/vqgan-gpt/vqgan_pretrained_models/s_32x32_8192.ckpt' \
# 	--resume '/scratch/eo41/vqgan-gpt/gpt_pretrained_models/s_gimel.pt' \
# 	--save_prefix 's_gimel'

# # # ####################### A #######################
# srun python -u /scratch/eo41/vqgan-gpt/train.py \
# 	--save_dir '/scratch/eo41/vqgan-gpt/gpt_pretrained_models' \
# 	--batch_size 6 \
# 	--gpt_config 'GPT_gimel' \
# 	--num_workers 16 \
# 	--print_freq 15000 \
# 	--optimizer ${OPTIMIZER} \
# 	--lr ${LR} \
# 	--seed ${SLURM_ARRAY_TASK_ID} \
# 	--data_path '/scratch/eo41/data/saycam/A_5fps_300s_{000000..000002}.tar' \
# 	--vqconfig_path '/scratch/eo41/vqgan-gpt/vqgan_pretrained_models/a_32x32_8192.yaml' \
# 	--vqmodel_path '/scratch/eo41/vqgan-gpt/vqgan_pretrained_models/a_32x32_8192.ckpt' \
# 	--resume '/scratch/eo41/vqgan-gpt/gpt_pretrained_models/a_gimel.pt' \
#  	--save_prefix 'a_gimel'

# ####################### Y #######################
srun python -u /scratch/eo41/vqgan-gpt/train.py \
	--save_dir '/scratch/eo41/vqgan-gpt/gpt_pretrained_models' \
	--batch_size 6 \
	--gpt_config 'GPT_gimel' \
	--num_workers 16 \
	--print_freq 15000 \
	--optimizer ${OPTIMIZER} \
	--lr ${LR} \
	--seed ${SLURM_ARRAY_TASK_ID} \
	--data_path '/scratch/eo41/data/saycam/Y_5fps_300s_{000000..000002}.tar' \
	--vqconfig_path '/scratch/eo41/vqgan-gpt/vqgan_pretrained_models/y_32x32_8192.yaml' \
	--vqmodel_path '/scratch/eo41/vqgan-gpt/vqgan_pretrained_models/y_32x32_8192.ckpt' \
	--resume '/scratch/eo41/vqgan-gpt/gpt_pretrained_models/y_gimel.pt' \
	--save_prefix 'y_gimel'

echo "Done"