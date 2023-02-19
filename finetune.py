import os
import time
import builtins
import argparse
import torch
import torch.distributed as dist
import numpy as np
import gptmodel
from torchvision.transforms import Compose, Resize, RandomCrop, ToTensor
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from utils import set_seed, load_config, load_vqgan, preprocess_vqgan, save_checkpoint

parser = argparse.ArgumentParser(description='Finetune a GPT on VQGAN encoded images')
parser.add_argument('--data_path', default="", type=str, help='data path')
parser.add_argument('--vqconfig_path', default="/scratch/eo41/vqgan-gpt/vqgan_pretrained_models/say_32x32_8192.yaml", type=str, help='vq config path')
parser.add_argument('--vqmodel_path', default="/scratch/eo41/vqgan-gpt/vqgan_pretrained_models/say_32x32_8192.ckpt", type=str, help=' vq model path')
parser.add_argument('--num_workers', default=8, type=int, help='number of data loading workers (default: 8)')
parser.add_argument('--seed', default=1, type=int, help='random seed')
parser.add_argument('--save_dir', default='', type=str, help='model save directory')
parser.add_argument('--save_prefix', default='model', type=str, help='model save name')
parser.add_argument('--save_freq', default=10, type=int, help='save checkpoint every this many epochs')
parser.add_argument('--gpt_config', default='GPT_bet', type=str, help='name of GPT config', choices=['GPT_alef', 'GPT_bet', 'GPT_gimel', 'GPT_dalet'])
parser.add_argument('--vocab_size', default=8192, type=int, help='vocabulary size')
parser.add_argument('--block_size', default=1023, type=int, help='context size')
parser.add_argument('--batch_size', default=32, type=int, help='batch size per gpu')
parser.add_argument('--lr', default=0.0005, type=float, help='learning rate')
parser.add_argument('--optimizer', default='Adam', choices=['Adam', 'SGD', 'ASGD'], help='optimizer')
parser.add_argument('--epochs', default=1000, type=int, help='number of finetuning epochs')
parser.add_argument('--resume', default='', type=str, help='Model path for resuming training')
parser.add_argument('--gpu', default=None, type=int)
parser.add_argument('--world-size', default=-1, type=int, help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int, help='node rank for distributed training')
parser.add_argument('--dist-url', default='env://', type=str, help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')
parser.add_argument('--local_rank', default=-1, type=int, help='local rank for distributed training')

args = parser.parse_args()
print(args)

# set random seed
set_seed(args.seed)

# DDP setting
if "WORLD_SIZE" in os.environ:
    args.world_size = int(os.environ["WORLD_SIZE"])
args.distributed = args.world_size > 1

if args.distributed:
    if args.local_rank != -1: # for torch.distributed.launch
        args.rank = args.local_rank
        args.gpu = args.local_rank
    elif 'SLURM_PROCID' in os.environ: # for slurm scheduler
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank)

    # suppress printing if not on master gpu
    if args.rank != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

# load vqgan model to encode images
vq_config = load_config(args.vqconfig_path, display=True)
vq_model = load_vqgan(vq_config, ckpt_path=args.vqmodel_path)
vq_model = vq_model.cuda(args.gpu)

# data pipeline
transform = Compose([Resize(288), RandomCrop(256), ToTensor()])
dataset = ImageFolder(args.data_path, transform)
sampler = DistributedSampler(dataset, seed=args.seed) if args.distributed else None
data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=(not args.distributed), num_workers=args.num_workers, pin_memory=True, sampler=sampler)
print('Data loaded: dataset contains {} images, and takes {} training iterations per epoch.'.format(len(dataset), len(data_loader)))

# set up model
mconf = gptmodel.__dict__[args.gpt_config](args.vocab_size, args.block_size)
model = gptmodel.GPT(mconf)

print('Running on {} GPUs total'.format(args.world_size))

if args.distributed:
    # For multiprocessing distributed, DDP constructor should always set the single device scope
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model.cuda(args.gpu)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    else:
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model)
else:
    model = torch.nn.DataParallel(model.cuda())

optimizer = torch.optim.__dict__[args.optimizer](model.parameters(), args.lr, weight_decay=0.0)

if os.path.isfile(args.resume):
    checkpoint = torch.load(args.resume, map_location='cpu')
    model.module.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print("=> loaded model weights and optimizer state at checkpoint '{}'".format(args.resume))
    del checkpoint
    torch.cuda.empty_cache()
else:
    print("=> no checkpoint loaded, will train from scratch")

# train model
model.train()
losses = []

for epoch in range(args.epochs):

    # the following is necessary to shuffle the order at the beginning of each epoch
    if args.distributed: 
        data_loader.sampler.set_epoch(epoch)

    start_time = time.time()

    for _, (images, _) in enumerate(data_loader):
        with torch.no_grad():
            images = preprocess_vqgan(images.cuda(args.gpu))
            _, _, [_, _, indices] = vq_model.encode(images)
            indices = indices.reshape(images.size(0), -1)

        # forward prop
        _, loss, _ = model(indices[:, :-1], indices[:, 1:])  # first output returns logits, last one returns unreduced losses
        losses.append(loss.item())

        # backprop and update the parameters
        model.zero_grad()
        loss.backward()
        optimizer.step()

    end_time = time.time()

    # log and save after every epoch
    train_loss = float(np.mean(losses))
    elapsed_time = end_time - start_time
    print('Epoch:', epoch, '|', 'Training loss:', train_loss, '|', 'Elapsed time:', elapsed_time)

    if epoch % args.save_freq == 0:
        # save trained model, etc.
        if args.distributed:
            if args.rank == 0:
                save_checkpoint(model, optimizer, train_loss, epoch, args.save_prefix, args.save_dir)
        else:
            save_checkpoint(model, optimizer, train_loss, epoch, args.save_prefix, args.save_dir)

    losses = []