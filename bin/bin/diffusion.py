import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import contextlib


class DDPM(nn.Module):
    def __init__(
        self,
        img_size,
        device,
        image_channels,
        n_diffusion_steps,
        init_beta,
        fin_beta
    ):
        super().__init__()

        self.img_size = img_size
        self.device = torch.device(device)
        
        self.image_channels = image_channels
        self.n_diffusion_steps = n_diffusion_steps
        self.init_beta = init_beta
        self.fin_beta = fin_beta

        self.beta = torch.linspace(
            self.init_beta,
            self.fin_beta,
            self.n_diffusion_steps,
            device=self.device,
        )
        self.alpha = 1 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

    @staticmethod
    def index(x, diffusion_step):
        return x[diffusion_step][:, None, None, None]

    @torch.no_grad()
    def sample_noise(self, batch_size):
        return torch.randn(
            size=(batch_size, self.image_channels, self.img_size, self.img_size),
            device=self.device,
        )

    @torch.no_grad()
    def sample_diffusion_step(self, batch_size):
        return torch.randint(
            0, self.n_diffusion_steps, size=(batch_size,), device=self.device,
        )

    @torch.no_grad()
    def batchify_diffusion_steps(self, diffusion_step_idx, batch_size):
        return torch.full(
            size=(batch_size,),
            fill_value=diffusion_step_idx,
            dtype=torch.long,
            device=self.device,
        )
    
    @torch.no_grad()
    def q_sample(self, x_0, t, nosie=None):

        alpha_bar_t = self.index(self.alpha_bar, diffusion_step=t)
        mean = (alpha_bar_t ** 0.5) * x_0
        var = 1 - alpha_bar_t

        if nosie is None:
            nosie = self.sample_noise(batch_size=x_0.size(0))

        x_t = mean + (var ** 0.5) * nosie

        return x_t

    def get_loss(self, network, x_0, y):
        t = self.sample_diffusion_step(batch_size=x_0.size(0))
        noise = self.sample_noise(batch_size=x_0.size(0))

        x_t = self.q_sample(x_0, t, noise)

        # with torch.autocast(device_type=self.device.type, dtype=torch.float32):
        pred_noise = network(
            noisy_image=x_t, diffusion_step=t, label=y,
        )

        return F.mse_loss(pred_noise, noise, reduction="mean")

    @torch.no_grad()
    def sample(**kwargs):
        raise NotImplementedError