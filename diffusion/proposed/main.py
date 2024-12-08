import os
from typing import Dict
import numpy as np

import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR100
from torchvision.utils import save_image

from trainer import DiffusionTrainer
from cfg import CFGModule
from network import UNet
from lr_scheduler import GradualWarmupScheduler
from var_scheduler import Scheduler
from path_manager import PathManager

def train(modelConfig):

    device = torch.device(modelConfig["device"])
    epochs = int(modelConfig["iters"] / (50000 / modelConfig['batch_size']))
    print ('epochs: ', epochs)
    # dataset
    dataset = CIFAR100(
        root='./CIFAR100', train=True, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]))
    dataloader = DataLoader(
        dataset, batch_size=modelConfig["batch_size"], shuffle=True, num_workers=4, drop_last=True, pin_memory=True)
    
    net_model = UNet(T=modelConfig["T"], num_labels=100, ch=modelConfig["channel"], ch_mult=modelConfig["channel_mult"],
                     num_res_blocks=modelConfig["num_res_blocks"], dropout=modelConfig["dropout"]).to(device)
    
    optimizer = torch.optim.AdamW(net_model.parameters(), lr=modelConfig["lr"], weight_decay=1e-4)
    cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=epochs, eta_min=0, last_epoch=-1)

    warmUpScheduler = GradualWarmupScheduler(optimizer=optimizer, multiplier=modelConfig["multiplier"],
                                             warm_epoch=epochs // 10, after_scheduler=cosineScheduler)
    
    var_scheduler = Scheduler(modelConfig["T"], modelConfig["beta_1"], modelConfig["beta_T"])
    module = CFGModule(net_model, var_scheduler)

    manager = PathManager(modelConfig)

    trainer = DiffusionTrainer(module, manager)

    trainer.train(dataloader, optimizer, modelConfig["iters"], warmUpScheduler)

def eval():
    pass

def main(model_config=None):
    modelConfig = {
        "state": "train", # or eval
        "iters": 40000,
        "batch_size": 80,
        "T": 500,
        "channel": 128,
        "channel_mult": [1, 2, 2, 2],
        "num_res_blocks": 2,
        "dropout": 0.15,
        "lr": 1e-4,
        "multiplier": 2.5,
        "beta_1": 1e-4,
        "beta_T": 0.028,
        "img_size": 32,
        "grad_clip": 1.,
        "device": "cuda",
        "w": 1.8,
        "training_load_weight": None,
        "sampled_dir": "./sampled_image/",
        "sampledNoisyImgName": "NoisyGuidenceImgs.png",
        "sampledImgName": "SampledGuidenceImgs.png",
        "nrow": 8
    }
    if model_config is not None:
        modelConfig = model_config
    if modelConfig["state"] == "train":
        train(modelConfig)
    else:
        eval(modelConfig)


if __name__ == '__main__':
    main()

