import torch
import torch.nn as nn

class DDPMScheduler(nn.Module):
    def __init__(self,
                 num_steps,
                 beta_1 = 1e-4,
                 beta_T = 0.02,
                 device = 'cuda'):
        super().__init__()
        self.num_steps = num_steps
        self.timesteps = list(range(1, self.num_steps))
        betas = torch.linspace(beta_1, beta_T, steps=num_steps, device=device)
        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        

class DDIMScheduler(DDPMScheduler):
    def __init__(self,
                 num_steps,
                 beta_1 = 1e-4,
                 beta_T = 0.02,
                 ddim_sampling_steps=50,
                 eta=0.0,
                 device='cuda'):
        super().__init__(num_steps=num_steps, beta_1=beta_1, beta_T=beta_T, device=device)
        self.eta = eta
        self.ddim_sampling_steps = ddim_sampling_steps
        # linear
        self.timesteps = list(range(1, self.num_steps, self.num_steps // self.ddim_sampling_steps))
