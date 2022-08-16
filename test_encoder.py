import yaml
import torch
import webdataset as wds

from omegaconf import OmegaConf
from vqmodel import VQModel
from torchvision.transforms import Compose, RandomResizedCrop, ToTensor

def preprocess(sample):
    return sample[0]

def preprocess_vqgan(x):
    x = 2.*x - 1.
    return x

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

data_path = "/scratch/eo41/data/saycam/SAY_5fps_300s_{000000..000009}.tar"
config_path = "/scratch/eo41/vqgan-gpt/vqgan_pretrained_models/say_32x32_8192.yaml"
model_path = "/scratch/eo41/vqgan-gpt/vqgan_pretrained_models/say_32x32_8192.ckpt"
batch_size_per_gpu = 1
num_workers = 8

configsaycam = load_config(config_path, display=True)
modelsaycam = load_vqgan(configsaycam, ckpt_path=model_path).to("cuda:0")

# data preprocessing
transform = Compose([RandomResizedCrop(256, scale=(0.4, 1)), ToTensor()])

dataset = (wds.WebDataset(data_path, resampled=True).shuffle(1000).decode("pil").to_tuple("jpg").map(preprocess).map(transform))
data_loader = wds.WebLoader(dataset, shuffle=False, batch_size=batch_size_per_gpu, num_workers=num_workers)

for it, images in enumerate(data_loader):
    images = torch.cat((images, images))
    images = preprocess_vqgan(images)
    images = images.to("cuda:0")
    print("images dtype, shape, max, min:", images.dtype, images.shape, images.max(), images.min())
    _, _, [_, _, indices] = modelsaycam.encode(images)
    indices = indices.reshape(2*batch_size_per_gpu, -1)
    print("indices dtype, shape, max, min:", indices.dtype, indices.shape, indices.max(), indices.min())
    #print(indices[0])
    #print(indices[1])
    print(torch.all(indices[0]==indices[1]))