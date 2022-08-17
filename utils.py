import os
import random
import numpy as np
import torch
import yaml
from omegaconf import OmegaConf
from vqmodel import VQModel


def load_config(config_path, display=False):
    config = OmegaConf.load(config_path)
    if display:
        print(yaml.dump(OmegaConf.to_container(config)))
        return config

def load_vqgan(config, ckpt_path=None):
    model = VQModel(**config.model.params)
    if ckpt_path is not None:
        sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        missing, unexpected = model.load_state_dict(sd, strict=False)
    return model.eval()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def preprocess(sample):
    return sample[0]

def preprocess_vqgan(x):
    x = 2.*x - 1.
    return x

def save_checkpoint(model, train_loss, it, model_name, save_dir):
    # DataParallel wrappers keep raw model object in .module attribute
    raw_model = model.module if hasattr(model, "module") else model
    
    # save everything we need
    save_str = 'model_{}_{}'.format(it, model_name)
    save_path = os.path.join(save_dir, save_str)
    print('Saving model to:', save_path)
    torch.save({'model_state_dict': raw_model.state_dict(), 'train_loss': train_loss}, save_path)