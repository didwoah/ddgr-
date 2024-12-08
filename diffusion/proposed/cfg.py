import torch
import torch.nn as nn

def extract(input, index):
    if index.ndim == 0:
        index = index.unsqueeze(0)
    index = index.long().to(input.device)
    
    return torch.gather(input, 0 ,index).view(index.shape[0], 1, 1, 1)

class CFGModule(nn.Module):
    def __init__(self, network, var_scheduler, cfg_factor = 1.8, device = 'cuda'):
        super().__init__()
        self.network = network
        self.var_scheduler = var_scheduler
        self.cfg_factor = cfg_factor
        self.device = device

    @torch.no_grad()
    def q_sample(self, x_0, t, noise=None):

        if noise is None:
            noise = torch.randn_like(x_0)

        alphas_prod_t = extract(self.var_scheduler.alphas_cumprod, t)
        x_t = torch.sqrt(alphas_prod_t) * x_0 + noise * torch.sqrt(1-alphas_prod_t)

        return x_t
    
    @torch.no_grad()
    def p_sample(self, x_t, t, y):

        if isinstance(t, int):
            t = torch.tensor([t]).to(self.device)

        eps_factor = (1 - extract(self.var_scheduler.alphas, t)) / (
            1 - extract(self.var_scheduler.alphas_cumprod, t)
        ).sqrt()

        noise = torch.randn_like(x_t)

        noise_factor = (
            (1 - extract(self.var_scheduler.alphas_cumprod, t-1)) / (1 - extract(self.var_scheduler.alphas_cumprod, t)) * extract(self.scheduler.betas, t)
        ).sqrt()
        t_expanded = t[:, None]
        noise_factor = torch.where(t_expanded>1, noise_factor, torch.zeros_like(noise_factor))

        eps_no_cond = self.network(x_t, t, None)
        eps_cond = self.network(x_t, t, y)
        cfg_eps = (1. + self.cfg_factor) * eps_cond - self.cfg_factor * eps_no_cond

        mean = (x_t - eps_factor * cfg_eps) / extract(self.var_scheduler.alphas, t).sqrt()

        x_t_prev = mean + noise_factor * noise

        return x_t_prev
    
    def loss_function(self, x_0, y):
        batch_size = x_0.shape[0]
        t = (
            torch.randint(0, self.var_scheduler.num_steps, size=(batch_size,), device = self.device)
            .long()
        )
        noise = torch.randn_like(x_0)
        x_t = self.q_sample(x_0, t, noise)
        eps_theta = self.network(x_t, t, y)
        loss = ((noise - eps_theta)**2.).mean()

        return loss
        
    @torch.no_grad()
    def sample(self, x_shape, y):
        # x_shape: (batch_size, c_dim, h_dim, w_dim)
        x = torch.randn(x_shape)
        batch_size = x_shape[0]

        for time_step in reversed(range(self.var_scheduler.num_steps)):
            t = torch.ones(size = (batch_size,)) * time_step
            t = t.to(self.device)
            x = self.p_sample(x, t, y)

        return x
    
    def save(self, save_path):
        torch.save(self.network.state_dict(), save_path)