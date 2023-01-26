import os
import argparse
import torch
import numpy as np
import gptmodel
from torch.nn import functional as F
from utils import load_config, load_vqgan, set_seed
from torchvision.utils import save_image


def top_k_logits(logits, k):
    v, _ = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float('Inf')
    return out

@torch.no_grad()
def sample(model, x, steps, temperature=1.0, sample=False, top_k=None):
    """
    take a conditioning sequence of indices in x (of shape (b,t)) and predict the next token in
    the sequence, feeding the predictions back into the model each time. Clearly the sampling
    has quadratic complexity unlike an RNN that is only linear, and has a finite context window
    of block_size, unlike an RNN that has an infinite context window.
    """
    
    block_size = model.get_block_size()
    model.eval()
    
    for k in range(steps):
        if k % 100 == 0:
            print('Step:', k)
        x_cond = x if x.size(1) <= block_size else x[:, -block_size:] # crop context if needed
        logits, _, _ = model(x_cond)
        # pluck the logits at the final step and scale by temperature
        logits = logits[:, -1, :] / temperature
        # optionally crop probabilities to only the top k options
        if top_k is not None:
            logits = top_k_logits(logits, top_k)
        # apply softmax to convert to probabilities
        probs = F.softmax(logits, dim=-1)
        # sample from the distribution or take the most likely
        if sample:
            ix = torch.multinomial(probs, num_samples=1)
        else:
            _, ix = torch.topk(probs, k=1, dim=-1)
        # append to the sequence and continue
        x = torch.cat((x, ix), dim=1)

    return x

def generate_samples(model, vocab_size, n_samples):
    # uniformly sample the first pixel
    counts = torch.ones(vocab_size)
    prob = counts / counts.sum()

    ## sample some generated images
    start_pixel = np.random.choice(np.arange(vocab_size), size=(n_samples, 1), replace=True, p=prob.numpy())
    start_pixel = torch.from_numpy(start_pixel)
    if torch.cuda.is_available():
        start_pixel = start_pixel.cuda()

    print('Starting sampling.')    
    pixels = sample(model, start_pixel, model.get_block_size(), temperature=0.96, sample=True, top_k=128)

    return pixels

def generate_samples_from_half(model, x, n_samples):
    print('Starting sampling.')
    all_pixels = []
    ctx_len = (model.get_block_size() + 1) // 2

    all_pixels.append(x)  # append the original images first
    for _ in range(n_samples-1):
        pixels = sample(model, x[:, :ctx_len], ctx_len, temperature=0.96, sample=True, top_k=128)
        all_pixels.append(pixels)

    return torch.cat(all_pixels)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Generate samples from a checkpointed GPT')
    parser.add_argument('--gpt_dir', default='/scratch/eo41/vqgan-gpt/gpt_pretrained_models', type=str, help='Directory storing the GPT model')
    parser.add_argument('--gpt_model', default='', type=str, help='Full name of the GPT model')
    parser.add_argument('--gpt_config', default='GPT_gimel', type=str, help='name of GPT config', choices=['GPT_alef', 'GPT_bet', 'GPT_gimel', 'GPT_dalet'])
    parser.add_argument('--vqconfig_path', default='/scratch/eo41/vqgan-gpt/vqgan_pretrained_models/say_32x32_8192.yaml', type=str, help='vq config path')
    parser.add_argument('--vqmodel_path', default='/scratch/eo41/vqgan-gpt/vqgan_pretrained_models/say_32x32_8192.ckpt', type=str, help=' vq model path')
    parser.add_argument('--n_samples', default=6, type=int, help='number of samples to generate')
    parser.add_argument('--data_path', default='/scratch/eo41/data/saycam/SAY_5fps_300s_{000000..000009}.tar', type=str, help='data path')
    parser.add_argument('--condition', default='free', type=str, help='Generation condition', choices=['free', 'cond'])
    parser.add_argument('--n_imgs', default=5, type=int, help='number of images')
    parser.add_argument('--seed', default=1, type=int, help='random seed')

    args = parser.parse_args()
    print(args)
    
    # set random seed
    set_seed(args.seed)

    mconf = gptmodel.__dict__[args.gpt_config](8192, 1023)
    model = gptmodel.GPT(mconf)

    # load the model
    print("Loading model")
    model_ckpt = torch.load(os.path.join(args.gpt_dir, args.gpt_model + '.pt'))
    model.load_state_dict(model_ckpt['model_state_dict'])

    if torch.cuda.is_available():
        model = model.cuda()

    # decode
    configsaycam = load_config(args.vqconfig_path, display=True)
    modelsaycam = load_vqgan(configsaycam, ckpt_path=args.vqmodel_path)
    modelsaycam = modelsaycam.cuda()

    if args.condition == 'free':
        # generate some samples unconditionally
        print("Generating unconditional samples")
        pixels = generate_samples(model, 8192, args.n_samples)
        print("pixels shape:", pixels.shape)
        n_totsamples = args.n_samples
    else:
        import webdataset as wds
        from torchvision.transforms import Compose, Resize, RandomCrop, ToTensor
        from utils import preprocess, preprocess_vqgan

        # data preprocessing
        transform = Compose([Resize(288), RandomCrop(256), ToTensor()])
        dataset = (wds.WebDataset(args.data_path, resampled=True).shuffle(10000, initial=10000).decode("pil").to_tuple("jpg").map(preprocess).map(transform))
        data_loader = wds.WebLoader(dataset, shuffle=False, batch_size=args.n_imgs, num_workers=4)

        for it, images in enumerate(data_loader):
            images = preprocess_vqgan(images)
            images = images.cuda()
            z, _, [_, _, indices] = modelsaycam.encode(images)
            indices = indices.reshape(args.n_imgs, -1)

            if it == 0:
                break

        # generate conditional samples
        print("Generating conditional samples")
        pixels = generate_samples_from_half(model, indices, args.n_samples)
        print("pixels shape:", pixels.shape)
        n_totsamples = args.n_samples * args.n_imgs

    z = modelsaycam.quantize.get_codebook_entry(pixels, (n_totsamples, 32, 32, 256))
    print("z shape:", z.shape)

    xmodel = modelsaycam.decode(z)
    print('xmodel shape:', xmodel.shape)

    if args.condition == "cond":
        xmodel[:, :, 126, :] = 1

    save_image(xmodel, "{}_{}_{}.png".format(args.condition, args.gpt_model, args.seed), nrow=5, padding=1, normalize=True)