import torch
import torch.nn as nn

class Scheduler(nn.Module):
    def __init__(self,
                 num_steps,
                 beta_1 = 1e-4,
                 beta_T = 0.02,
                 device = 'cuda'):
        super().__init__()
        self.num_steps = num_steps
        betas = torch.linspace(beta_1, beta_T, steps=num_steps, device=device)
        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
