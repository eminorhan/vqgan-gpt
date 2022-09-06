import os
import sys
import argparse
import webdataset as wds
from torchvision.transforms import Compose, Resize, ToTensor
from torchvision.utils import save_image

sys.path.insert(0, '/scratch/eo41/vqgan-gpt')
from utils import preprocess, preprocess_vqgan

def identity(x):
    return x

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Reconstruction fidelity tests')
    parser.add_argument('--data_path', default="/scratch/eo41/data/saycam/SAY_5fps_300s_{000000..000009}.tar", type=str, help='data path')
    parser.add_argument('--batch_size_per_gpu', default=1, type=int, help='batch size per gpu')
    parser.add_argument('--num_workers', default=4, type=int, help='number of data loading workers (default: 4)')
    parser.add_argument('--n_samples', default=100, type=int, help='number of images to sample')
    parser.add_argument('--save_path', default="/scratch/eo41/vqgan-gpt/tests/shuffle_test", type=str, help='directory to store original and reconstruction images')

    args = parser.parse_args()
    print(args)

    # data preprocessing
    transform = Compose([Resize(256), ToTensor()])
    dataset = (wds.WebDataset(args.data_path, resampled=True).shuffle(10000, initial=10000).decode("pil").to_tuple("jpg", "cls").map_tuple(transform, identity))
    # dataset = (wds.WebDataset(args.data_path, resampled=True).shuffle(50000, initial=50000).decode("pil").to_tuple("jpg").map(preprocess).map(transform))
    data_loader = wds.WebLoader(dataset, shuffle=False, batch_size=args.batch_size_per_gpu, num_workers=args.num_workers)


    for it, (images, cls) in enumerate(data_loader):

        images = preprocess_vqgan(images)
        images = images.to("cuda:0")
        save_image(images, os.path.join(args.save_path, "image_{:04d}_{:04d}.jpeg".format(it, int(cls))), normalize=True)

        if it == args.n_samples:
            break