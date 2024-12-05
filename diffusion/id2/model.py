import os
import copy
import random

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

        self.device = next(diffusion_network.parameters()).device


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

    
    def train(self, dataloader, epochs = 200, device="cpu", patience=5):
        self.diffusion_network.to(device)

        progress_bar = tqdm(range(epochs), desc="Training Progress", leave=True)
        best_loss = 0.0
        best_model = copy.deepcopy(self.diffusion_network)
        no_improvement_epochs = 0

        for epoch in progress_bar:
            self.diffusion_network.train()
            running_loss = 0.0
            total = 0

            for inputs, labels in dataloader:
                inputs, labels = inputs.to(device), labels.to(device)

                self.diffusion_optimizer.zero_grad()

                loss = self.diffusion_method.get_unet_loss(
                    diffusion_network=self.diffusion_network,
                    ori_image=inputs, 
                    label=labels)

                loss.backward()

                self.diffusion_optimizer.step()

                running_loss += loss.item()

                total += labels.size(0)

            avg_loss = running_loss / len(dataloader)

            # Update progress bar
            progress_bar.set_postfix(epoch=epoch + 1, loss=f"{avg_loss:.4f}")

            # Early stopping check
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_model = copy.deepcopy(self.diffusion_network)
                no_improvement_epochs = 0  # Reset counter
            else:
                no_improvement_epochs += 1

            if no_improvement_epochs >= patience:
                print(f"\nEarly stopping triggered. Best loss: {best_loss:.2f}%")
                break

        print("Training completed.")
        return best_model
    
    
