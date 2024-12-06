import os

import random

import torch
from torch import nn
import torch.nn.functional as F

from itertools import cycle
from torch.optim.optimizer import Optimizer

from .method import Id12Method
from .network import UNet
from tqdm import tqdm

from utils import sample

class Id12(nn.Module):
    def __init__(self, 
                 diffusion_network: UNet,
                 prev_diffusion_network: UNet,
                 diffusion_method: Id12Method,
                 diffusion_optimizer: Optimizer,
                 
                 classifier_network,
                 
                 label_pool,
                 prev_label_pool,
                 distillation_ratio=0.5,
                 distillation_label_ratio=0.5,
                 distillation_trial=5):
        super().__init__()

        self.diffusion_network = diffusion_network
        self.diffusion_method = diffusion_method
        self.diffusion_optimizer = diffusion_optimizer

        self.prev_diffusion_network = prev_diffusion_network
        
        self.classifier_network = classifier_network

        self.distillation_ratio = distillation_ratio
        self.distillation_label_ratio = distillation_label_ratio
        self.distillation_trial = distillation_trial
        
        self.label_pool = label_pool                # current labels
        self.prev_label_pool = prev_label_pool      # previous labels


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
    
    # def classifier_noise_train_a_batch(self, x, y) -> torch.Tensor:
    #     rand_diffusion_step = self.model.sample_diffusion_step(batch_size=x.size(0))
    #     rand_noise = self.model.sample_noise(batch_size=x.size(0))
    #     noisy_image = self.model.perform_diffusion_process(
    #         ori_image=x,
    #         diffusion_step=rand_diffusion_step,
    #         rand_noise=rand_noise,
    #     )

    #     self.classifier_optimizer.zero_grad()

    #     loss = self.model.classifier.get_loss(noisy_image, rand_diffusion_step, y)

    #     loss.backward()

    #     torch.nn.utils.clip_grad_norm_(self.model.classifier.parameters(), max_norm=1.0)

    #     self.classifier_optimizer.step()

    #     return loss


    def get_distillation_labels(self):
        
        assert self.prev_label_pool is not None
        
        n_selection = max(int(self.prev_label_pool*self.distillation_label_ratio), 1)

        selected_y = sample(self.prev_label_pool, n_selection)
        # mask = torch.tensor([False] * y.shape[0]).to(self.device)
        # for selected in selected_y:
        #     mask = mask | (y == selected)
        # if not mask.any():
        #     mask[0] = True
            
        # selected_x = x[mask]

        return selected_y


    def distillate(self, x, device='cpu'):
        selected_y = self.get_distillation_labels()

        with torch.no_grad():
            prev_errors = self.get_classify_errors(x, selected_y, self.prev_diffusion_network, device=device)
            prev_dist = self.get_distribution_of_class(selected_y, prev_errors, device)

        errors = self.get_classify_errors(x, selected_y, self.diffusion_network, device)
        dist = self.get_distribution_of_class(selected_y, errors, device)
        
        loss = F.kl_div(dist.log(), prev_dist.detach(), reduction='batchmean')
        
        return loss


    def train_a_batch(self, x, y, x_=None, y_=None, importance_of_new_task=.5, device='cpu'):
        assert x_ is None or x.size() == x_.size()
        assert y_ is None or y.size() == y_.size()

        # noise_train_loss = self.classifier_noise_train_a_batch(x, y)

        self.diffusion_optimizer.zero_grad()
               
        new_task_loss = self.diffusion_method.get_unet_loss(self.diffusion_network, x, y)
        if x_ is not None and y_ is not None:
            old_task_loss = self.diffusion_method.get_unet_loss(self.diffusion_network, x_, y_)

            loss = (
                importance_of_new_task * new_task_loss +
                (1-importance_of_new_task) * old_task_loss
            )
        else:
            loss = new_task_loss

        if self.prev_diffusion_network is not None:
            ditillation_loss = self.distillate(x, device=device)

            alpha = self.distillation_ratio
            loss = (1-alpha)*loss + alpha*ditillation_loss

        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.diffusion_network.parameters(), max_norm=1.0)

        self.diffusion_optimizer.step()

        return loss
    
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

            # train with distillation
            loss = self.train_a_batch(x=image, y=labels, device=device)

            running_loss += loss.item()

            total += labels.size(0)

            avg_loss = running_loss / (iter+1)

            # Update progress bar
            progress_bar.set_postfix(iter=iter + 1, loss=f"{avg_loss:.4f}")


        print("Training completed.")
    

    def classify(self, x):
        errors = self.get_classify_errors(x, self.label_pool, self.diffusion_network)

        dist = self.get_distribution_of_class(errors)

        pred, _ = min(enumerate(dist), key=lambda x: x[1])

        return pred
    
    def get_distribution_of_class(self, label_pool, errors, device):
        mean_errors = torch.tensor([0]).to(device)
        for label in sorted(label_pool):
            mean = torch.mean(errors[label]).unsqueeze(0)
            mean_errors = torch.cat([mean_errors, -mean], dim=0)
        
        return F.softmax(mean_errors[1:])
    
    def get_classify_errors(self, x, label_pool, diffusion_model, device):
        errors = {}
        trial = self.distillation_trial
        for _ in range(trial):
            rand_diffusion_step = self.diffusion_method.sample_diffusion_step(batch_size=x.size(0))
            rand_noise = self.diffusion_method.sample_noise(batch_size=x.size(0)).detach()
            noisy_image = self.diffusion_method.perform_diffusion_process(
                ori_image=x,
                diffusion_step=rand_diffusion_step,
                rand_noise=rand_noise,
            ).detach()

            for label in label_pool:
                y = torch.tensor([label]*x.shape[0]).to(device)
                eps_th = diffusion_model(noisy_image, rand_diffusion_step, y)

                error = (eps_th - rand_noise)**2    # L2

                if label in errors.keys():
                    errors[label] = torch.cat([errors[label], error], dim=0)
                else:
                    errors[label] = error
        
        return errors
    
