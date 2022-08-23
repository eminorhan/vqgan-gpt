import argparse
import yaml
import torch
import webdataset as wds
import numpy as np
from omegaconf import OmegaConf
from vqmodel import VQModel
from torchvision.transforms import Compose, RandomResizedCrop, ToTensor
from PIL import Image, ImageDraw

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

def custom_to_pil(x):
  x = x.detach().cpu()
  x = torch.clamp(x, -1., 1.)
  x = (x + 1.)/2.
  x = x.permute(1, 2, 0).numpy()
  x = (255 * x).astype(np.uint8)
  x = Image.fromarray(x)
  if not x.mode == "RGB":
    x = x.convert("RGB")
  return x

def stack_reconstructions(input, x0, titles=[]):
  assert input.size == x0.size
  w, h = input.size[0], input.size[1]
  img = Image.new("RGB", (2*w, h))
  img.paste(input, (0, 0))
  img.paste(x0, (1*w, 0))

  for i, title in enumerate(titles):
    ImageDraw.Draw(img).text((i*w, 0), f'{title}', (255, 255, 255))  # coordinates, text, color, font
  img.save("input_reconstruction_example.jpg")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Encoder-decoder tests')
    parser.add_argument('--data_path', default="/scratch/eo41/data/saycam/SAY_5fps_300s_{000000..000009}.tar", type=str, help='data path')
    parser.add_argument('--config_path', default="/scratch/eo41/vqgan-gpt/vqgan_pretrained_models/say_32x32_8192.yaml", type=str, help='config path')
    parser.add_argument('--model_path', default="/scratch/eo41/vqgan-gpt/vqgan_pretrained_models/say_32x32_8192.ckpt", type=str, help='model path')
    parser.add_argument('--batch_size_per_gpu', default=2, type=int, help='batch size per gpu')
    parser.add_argument('--num_workers', default=4, type=int, help='number of data loading workers (default: 4)')
    parser.add_argument('--test_reconstruct', default=True, type=bool, help='whether to test a reconstruction example')

    args = parser.parse_args()

    configsaycam = load_config(args.config_path, display=True)
    modelsaycam = load_vqgan(configsaycam, ckpt_path=args.model_path).to("cuda:0")

    # data preprocessing
    transform = Compose([RandomResizedCrop(256, scale=(0.4, 1)), ToTensor()])
    dataset = (wds.WebDataset(args.data_path, resampled=True).shuffle(1000).decode("pil").to_tuple("jpg").map(preprocess).map(transform))
    data_loader = wds.WebLoader(dataset, shuffle=False, batch_size=args.batch_size_per_gpu, num_workers=args.num_workers)

    for it, images in enumerate(data_loader):
        images = preprocess_vqgan(images)
        images = images.to("cuda:0")
        print("images dtype, shape, max, min:", images.dtype, images.shape, images.max(), images.min())
        z, _, [_, _, indices] = modelsaycam.encode(images)
        indices = indices.reshape(args.batch_size_per_gpu, -1)
        print("indices dtype, shape, max, min:", indices.dtype, indices.shape, indices.max(), indices.min())
        print("z dtype, shape, max, min:", z.dtype, z.shape, z.max(), z.min())

        if args.test_reconstruct:
            # test if we can reconstruct original image
            xmodel = modelsaycam.decode(z)
            print('xmodel shape:', xmodel.shape)
            imgstack = stack_reconstructions(custom_to_pil(images[0]), custom_to_pil(xmodel[0]), titles=["Input", "Reconstruction"])

        if it == 0:
            break