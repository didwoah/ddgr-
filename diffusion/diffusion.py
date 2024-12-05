import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import contextlib


class DDPM():
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

    def sample_noise(self, batch_size):
        return torch.randn(
            size=(batch_size, self.image_channels, self.img_size, self.img_size),
            device=self.device,
        )

    def sample_diffusion_step(self, batch_size):
        return torch.randint(
            0, self.n_diffusion_steps, size=(batch_size,), device=self.device,
        )

    def batchify_diffusion_steps(self, diffusion_step_idx, batch_size):
        return torch.full(
            size=(batch_size,),
            fill_value=diffusion_step_idx,
            dtype=torch.long,
            device=self.device,
        )

    def perform_diffusion_process(self, ori_image, diffusion_step, rand_noise=None):
        alpha_bar_t = self.index(self.alpha_bar, diffusion_step=diffusion_step)
        mean = (alpha_bar_t ** 0.5) * ori_image
        var = 1 - alpha_bar_t
        if rand_noise is None:
            rand_noise = self.sample_noise(batch_size=ori_image.size(0))
        noisy_image = mean + (var ** 0.5) * rand_noise
        return noisy_image

    def get_unet_loss(self, diffusion_network, ori_image, label):
        rand_diffusion_step = self.sample_diffusion_step(batch_size=ori_image.size(0))
        rand_noise = self.sample_noise(batch_size=ori_image.size(0))
        noisy_image = self.perform_diffusion_process(
            ori_image=ori_image,
            diffusion_step=rand_diffusion_step,
            rand_noise=rand_noise,
        )
        # with torch.autocast(device_type=self.device.type, dtype=torch.float32):
        pred_noise = diffusion_network(
            noisy_image=noisy_image, diffusion_step=rand_diffusion_step, label=label,
        )
        return F.mse_loss(pred_noise, rand_noise, reduction="mean")

    def sample(**kwargs):
        raise NotImplementedError