import torch
import torch.nn as nn

import os
from tqdm import tqdm
import itertools
import numpy as np

from diffusion.proposed.cfg import CFGModule

class DiffusionTrainer():
    def __init__(
            self,
            module: CFGModule,
            manager,
            kd_module = None,
            kd_factor = 0.,
            device = 'cuda'
            ):
        self.module = module
        self.kd_module = kd_module
        self.manager = manager
        self.kd_factor = kd_factor
        self.device = device

    def train(self, loader, optimizer, iters, train_scheduler = None):
        
        pbar = tqdm(range(iters), desc="Diffusion Training", unit="iter")
        iters_per_epoch = len(loader)
        loader_cycle = itertools.cycle(loader)

        for iteration in pbar:
            x, _, y = next(loader_cycle)
            x = x.to(self.device)
            y = y.to(self.device)

            if np.random.rand() < 0.1:
                y = torch.fill(y, 100)

            optimizer.zero_grad()
            loss = self.module.loss_function(x, y) 

            if self.kd_module is not None:
                loss += self.kd_factor * self.kd_module.loss_function(x)

            loss.backward()
            optimizer.step()

            if train_scheduler and ((iteration + 1) % iters_per_epoch == 0):
                train_scheduler.step()

            pbar.set_postfix({
                "Iteration": iteration + 1,
                "Loss": loss.item(),
                "LR": optimizer.param_groups[0]['lr']
            })

            if iteration >= iters - 1:
                break

        save_path = self.path_manager.get_model_path('generator')
        self.module.save(save_path)

