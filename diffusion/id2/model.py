import os
import copy
import random
from itertools import cycle

import torch
from torch import nn
from torch.optim.optimizer import Optimizer

from tqdm import tqdm

from .method import Id2Method
from .network import UNet

class Id2():
    def __init__(self, 
                 diffusion_network: UNet,
                 diffusion_method: Id2Method,
                 diffusion_optimizer: Optimizer,
                 
                 classifier_network):
        super().__init__()

        self.diffusion_network = diffusion_network
        self.diffusion_method = diffusion_method
        self.diffusion_optimizer = diffusion_optimizer

        self.classifier_network = classifier_network


    def sample(self, size, y):
        self.diffusion_network.eval()
        self.classifier_network.eval()

        samples = self.diffusion_method.sample(
            self.diffusion_network, 
            self.classifier_network, 
            size, y)
        
        self.diffusion_network.train()
        self.classifier_network.train()

        return samples
    

    def get_noisy_image(self, x):
        rand_diffusion_step = self.diffusion_method.sample_diffusion_step(batch_size=x.size(0))
        rand_noise = self.diffusion_method.sample_noise(batch_size=x.size(0))
        noisy_image = self.diffusion_method.perform_diffusion_process(
            ori_image=x,
            diffusion_step=rand_diffusion_step,
            rand_noise=rand_noise,
        )
        return noisy_image

    
    def train(self, dataloader, iters = 50000, device="cpu"):
        self.diffusion_network.to(device)

        progress_bar = tqdm(range(iters), desc="Training Progress", leave=True)

        dataloader = cycle(dataloader)

        for iter in progress_bar:
            self.diffusion_network.train()
            running_loss = 0.0
            total = 0

            image, _, new_label = next(dataloader)

            image, labels = image.to(device), new_label.to(device)

            self.diffusion_optimizer.zero_grad()

            loss = self.diffusion_method.get_unet_loss(
                diffusion_network=self.diffusion_network,
                ori_image=image, 
                label=labels)

            loss.backward()

            self.diffusion_optimizer.step()

            running_loss += loss.item()

            total += labels.size(0)

            avg_loss = running_loss / (iter+1)

            # Update progress bar
            progress_bar.set_postfix(iter=iter + 1, loss=f"{avg_loss:.4f}")


        print("Training completed.")
    
    
