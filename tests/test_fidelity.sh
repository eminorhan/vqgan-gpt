#!/bin/bash

#SBATCH --account=cds
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=200GB
#SBATCH --time=06:00:00
#SBATCH --array=0
#SBATCH --job-name=test_fidelity
#SBATCH --output=test_fidelity_%A_%a.out

module purge
module load cuda/11.3.1

#python -u /scratch/eo41/vqgan-gpt/tests/test_fidelity.py --vq_model "say_32x32_8192" --n_samples 9999
#python -u /scratch/eo41/vqgan-gpt/tests/test_fidelity.py --vq_model "s_32x32_8192" --n_samples 9999
#python -u /scratch/eo41/vqgan-gpt/tests/test_fidelity.py --vq_model "a_32x32_8192" --n_samples 9999
#python -u /scratch/eo41/vqgan-gpt/tests/test_fidelity.py --vq_model "y_32x32_8192" --n_samples 9999
#python -u /scratch/eo41/vqgan-gpt/tests/test_fidelity.py --vq_model "openimages_32x32_8192" --n_samples 9999 --gumbel
#python -u /scratch/eo41/vqgan-gpt/tests/test_fidelity.py --vq_model "imagenet_16x16_16384" --n_samples 9999

python -u /scratch/eo41/vqgan-gpt/tests/test_fidelity.py --data_path "/scratch/eo41/data/imagenet_val/imagenet_val_000000.tar" --save_path "/scratch/eo41/vqgan-gpt/tests/imagenet_rec_fid" --vq_model "say_32x32_8192" --n_samples 9999
python -u /scratch/eo41/vqgan-gpt/tests/test_fidelity.py --data_path "/scratch/eo41/data/imagenet_val/imagenet_val_000000.tar" --save_path "/scratch/eo41/vqgan-gpt/tests/imagenet_rec_fid" --vq_model "s_32x32_8192" --n_samples 9999
python -u /scratch/eo41/vqgan-gpt/tests/test_fidelity.py --data_path "/scratch/eo41/data/imagenet_val/imagenet_val_000000.tar" --save_path "/scratch/eo41/vqgan-gpt/tests/imagenet_rec_fid" --vq_model "a_32x32_8192" --n_samples 9999
python -u /scratch/eo41/vqgan-gpt/tests/test_fidelity.py --data_path "/scratch/eo41/data/imagenet_val/imagenet_val_000000.tar" --save_path "/scratch/eo41/vqgan-gpt/tests/imagenet_rec_fid" --vq_model "y_32x32_8192" --n_samples 9999
python -u /scratch/eo41/vqgan-gpt/tests/test_fidelity.py --data_path "/scratch/eo41/data/imagenet_val/imagenet_val_000000.tar" --save_path "/scratch/eo41/vqgan-gpt/tests/imagenet_rec_fid" --vq_model "openimages_32x32_8192" --n_samples 9999 --gumbel
python -u /scratch/eo41/vqgan-gpt/tests/test_fidelity.py --data_path "/scratch/eo41/data/imagenet_val/imagenet_val_000000.tar" --save_path "/scratch/eo41/vqgan-gpt/tests/imagenet_rec_fid" --vq_model "imagenet_16x16_16384" --n_samples 9999

echo "Done"
