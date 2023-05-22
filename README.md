# VQGAN-GPT

This repository can be used to train-finetune GPT models on VQGAN encoded discrete latents.

## Usage examples

### Training 
To train a GPT model on your data, use [`train.py`](https://github.com/eminorhan/vqgan-gpt/blob/master/train.py): 
```python
python -u train.py \
	--batch_size 6 \
	--gpt_config 'GPT_gimel' \
	--num_workers 16 \
	--print_freq 15000 \
	--lr 0.0003\
	--data_path DATA_PATH \
	--vqconfig_path VQCONFIG_PATH \
	--vqmodel_path VQMODEL_PATH \
	--resume '' \
	--save_dir SAVE_DIR \
	--save_prefix INFORMATIVE_SAVE_PREFIX
```
where `vqconfig_path` is the path to the config file for the VQGAN component of the model (config files for all models are available inside the [`vqgan_pretrained_models`](https://github.com/eminorhan/vqgan-gpt/tree/master/vqgan_pretrained_models) directory), `vqmodel_path` is the path to the pretrained VQGAN model, and `resume` is the GPT checkpoint location if starting from a saved checkpoint. Note that the training code uses the [`webdataset`](https://github.com/webdataset/webdataset) interface to feed the data into the model.

### Finetuning
Finetuning works similarly. To finetune a pretrained GPT model on your data, use [`finetune.py`](https://github.com/eminorhan/vqgan-gpt/blob/master/finetune.py): 
```python
python -u finetune.py \
	--batch_size 8 \
	--gpt_config 'GPT_gimel' \
	--num_workers 16 \
	--lr 0.0003\
	--data_path DATA_PATH \
	--vqconfig_path VQCONFIG_PATH \
	--vqmodel_path VQMODEL_PATH \
	--resume GPT_CHECKPOINT_TO_START_FROM \
	--save_dir SAVE_DIR \
	--save_prefix INFORMATIVE_SAVE_PREFIX
```
This uses the standard `torch`-`torchvision` data loading interface to feed the data into the model.