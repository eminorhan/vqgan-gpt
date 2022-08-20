import argparse
import torch
import numpy as np
from gptmodel import GPT, GPTConfig
from torch.nn import functional as F
from utils import load_config, load_vqgan
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
    # to sample we also have to technically "train" a separate model for the first token in the sequence
    # we are going to do so below simply by calculating and normalizing the histogram of the first token
    counts = torch.ones(vocab_size)  # start counts as 1 not zero, this is called "smoothing"
    prob = counts / counts.sum()

    ## sample some generated images
    start_pixel = np.random.choice(np.arange(vocab_size), size=(n_samples, 1), replace=True, p=prob.numpy())
    start_pixel = torch.from_numpy(start_pixel)
    if torch.cuda.is_available():
        start_pixel = start_pixel.cuda()

    print('Starting sampling.')    
    pixels = sample(model, start_pixel, model.get_block_size(), temperature=0.996, sample=True, top_k=128)

    return pixels

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Generate samples from a checkpointed GPT')
    parser.add_argument('--model_cache', default="/scratch/eo41/vqgan-gpt/gpt_pretrained_models/model_3200_24l_8h_512e_96b_0.0005lr_Adamo_0s.pt", type=str, help='Cache path for the GPT model')
    parser.add_argument('--vqconfig_path', default="/scratch/eo41/vqgan-gpt/vqgan_pretrained_models/say_32x32_8192.yaml", type=str, help='vq config path')
    parser.add_argument('--vqmodel_path', default="/scratch/eo41/vqgan-gpt/vqgan_pretrained_models/say_32x32_8192.ckpt", type=str, help=' vq model path')
    parser.add_argument('--n_samples', default=16, type=int, help='number of samples to generate')
    parser.add_argument('--filename', default='', type=str, help='file name to save')
    parser.add_argument('--img_dir', default='', type=str, help='directory of test images')

    args = parser.parse_args()
    print(args)

    ## set up model (TODO: better way to handle the model config)
    mconf = GPTConfig(8192, 1023, embd_pdrop=0.0, resid_pdrop=0.0, attn_pdrop=0.0, n_layer=24, n_head=8, n_embd=512)
    model = GPT(mconf)

    # load the model
    print("Loading model")
    model_ckpt = torch.load(args.model_cache)
    model.load_state_dict(model_ckpt['model_state_dict'])

    if torch.cuda.is_available():
        model = model.cuda()

    # generate some samples unconditionally
    print("Generating unconditional samples")
    pixels = generate_samples(model, 8192, args.n_samples)
    print("pixels shape:", pixels.shape)
    # pixels = pixels.view(args.n_samples, 32, 32)

    # decode
    configsaycam = load_config(args.vqconfig_path, display=True)
    modelsaycam = load_vqgan(configsaycam, ckpt_path=args.vqmodel_path)
    modelsaycam = modelsaycam.cuda()

    z = modelsaycam.quantize.get_codebook_entry(pixels, (args.n_samples, 32, 32, 256))
    print("z shape:", z.shape)

    xmodel = modelsaycam.decode(z)
    print('xmodel shape:', xmodel.shape)
    save_image(xmodel, "samples.pdf", nrow=4, padding=1, normalize=True)