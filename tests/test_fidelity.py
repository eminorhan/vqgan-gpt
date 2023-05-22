import os
import sys
import argparse
import webdataset as wds
from torchvision.transforms import Compose, Resize, RandomCrop, ToTensor
from torchvision.utils import save_image
from torch_fidelity import calculate_metrics

sys.path.insert(0, '/scratch/eo41/vqgan-gpt')
from utils import load_config, load_vqgan, preprocess, preprocess_vqgan

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Reconstruction fidelity tests')
    parser.add_argument('--data_path', default="/scratch/eo41/data/saycam/SAY_5fps_300s_{000000..000009}.tar", type=str, help='data path')
    parser.add_argument('--vq_path', default="/scratch/eo41/vqgan-gpt/vqgan_pretrained_models", type=str, help='directory storing VQ config and checkpoint')
    parser.add_argument('--batch_size_per_gpu', default=1, type=int, help='batch size per gpu')
    parser.add_argument('--num_workers', default=4, type=int, help='number of data loading workers (default: 4)')
    parser.add_argument('--n_samples', default=100, type=int, help='number of images to sample')
    parser.add_argument('--save_path', default="/scratch/eo41/vqgan-gpt/tests/saycam_rec_fid", type=str, help='directory to store original and reconstruction images')
    parser.add_argument('--vq_model', default="say_32x32_8192", type=str, help='VQ model name', choices=['say_32x32_8192', 's_32x32_8192', 'a_32x32_8192', 'y_32x32_8192', 'openimages_32x32_8192', 'imagenet_16x16_16384'])
    parser.add_argument('--gumbel', default=False, action='store_true', help='Gumbel VQ?')

    args = parser.parse_args()
    print(args)

    img_dir = os.path.join(args.save_path, args.vq_model, 'imgs')
    rec_dir = os.path.join(args.save_path, args.vq_model, 'recs')

    os.mkdir(img_dir)
    os.mkdir(rec_dir)

    configsaycam = load_config(os.path.join(args.vq_path, args.vq_model + ".yaml"), display=True)
    modelsaycam = load_vqgan(configsaycam, ckpt_path=os.path.join(args.vq_path, args.vq_model + ".ckpt"), gumbel=args.gumbel).to("cuda:0")

    # data preprocessing
    transform = Compose([Resize(288), RandomCrop(256), ToTensor()])
    dataset = (wds.WebDataset(args.data_path, resampled=True).shuffle(10000).decode("pil").to_tuple("jpg").map(preprocess).map(transform))
    data_loader = wds.WebLoader(dataset, shuffle=False, batch_size=args.batch_size_per_gpu, num_workers=args.num_workers)

    for it, images in enumerate(data_loader):
        images = preprocess_vqgan(images)
        images = images.to("cuda:0")
        z, _, [_, _, indices] = modelsaycam.encode(images)
        indices = indices.reshape(args.batch_size_per_gpu, -1)

        # test if we can reconstruct original image
        xmodel = modelsaycam.decode(z)

        save_image(images, os.path.join(img_dir, "image_{:04d}.jpeg".format(it)), normalize=True)
        save_image(xmodel, os.path.join(rec_dir, "recon_{:04d}.jpeg".format(it)), normalize=True)
        
        if it == args.n_samples:
            break

    metrics_dict = calculate_metrics(input1=img_dir, input2=rec_dir, isc=True, fid=True)
    print(metrics_dict)

    # save to file
    f = open(args.vq_model + ".txt", "w")
    f.write(str(metrics_dict))
    f.close()