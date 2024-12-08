# References:
    # https://github.com/openai/guided-diffusion/blob/22e0df8183507e13a7813f8d38d51b072ca1e67c/scripts/classifier_sample.py

import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import contextlib

from const import EPSILON

from diffusion.diffusion import DDPM

class Id12Method(DDPM):

    def __init__(
        self,
        img_size,
        device,
        lambda_cg,
        lambda_cfg,
        cfg_uncondition_train_ratio=0.2,
        classifier_scale=1,
        image_channels=3,
        n_diffusion_steps=1000,
        init_beta=0.0001,
        fin_beta=0.02,
    ):
        super().__init__(
            img_size=img_size,
            device=device,
            image_channels=image_channels,
            n_diffusion_steps=n_diffusion_steps,
            init_beta=init_beta,
            fin_beta=fin_beta,
        )

        self.lambda_cg = lambda_cg
        self.classifier_scale = classifier_scale

        self.lambda_cfg = lambda_cfg
        self.cfg_uncondition_train_ratio = cfg_uncondition_train_ratio


    @torch.enable_grad()
    def get_classifier_grad(self, classifier_network, alpha_bar_t, pred_noise_uncond, noisy_image, label):
        noisy_image.requires_grad = True
        
        # out = classifier_network(noisy_image)   # noisy_image = x_t  -->   We trained the classifier network using x_0
        
        # using tweedie's formula
        x_0 = (noisy_image - (1 - alpha_bar_t).sqrt() * pred_noise_uncond) / alpha_bar_t.sqrt()
        
        out = classifier_network(x_0)   # $\nabla_{x_{t}}\log{p_{\phi}}(y | x_0) not x_t
        
        # But, we can approximate p(y | x_t)  \approx  p(y | x_0)   from Diffusion Posterior Sampling Theorem 1.
        # At this case, forward measurement operator can be nonlinear operator. --> classifier p(y | x) is nonlinear
        
        log_prob = F.log_softmax(out, dim=-1)
            
        selected = log_prob[torch.arange(log_prob.size(0), dtype=torch.long, device=self.device), label.long()]

        # "$\nabla_{x_{t}}\log{p_{\phi}}(y \vert x)$"
        return torch.autograd.grad(outputs=selected.sum(), inputs=noisy_image)[0]


    @torch.no_grad()
    def p_sample(self, gen_net, cls_net, x_t, t, y):

        diffusion_step = self.batchify_diffusion_steps(
            diffusion_step_idx=t, batch_size=x_t.size(0),
        )

        alpha_t = self.index(self.alpha, diffusion_step=diffusion_step)
        beta_t = self.index(self.beta, diffusion_step=diffusion_step)
        alpha_bar_t = self.index(self.alpha_bar, diffusion_step=diffusion_step)
        
        x_t = x_t.detach().clone()
        

        with torch.inference_mode():
            pred_noise_cond = gen_net(
                noisy_image=x_t.detach(), diffusion_step=diffusion_step, label=y,
            )

            pred_noise_uncond = gen_net(
                noisy_image=x_t.detach(), diffusion_step=diffusion_step, label=None
            )
            
        # approximating \nabla_{x_t} log p(y | x_t)  \approx   \nabla_{x_t} log p(y | x_0)
        # grad = self.get_classifier_grad(
        #     noisy_image=x_t,
        #     alpha_bar_t=alpha_bar_t,
        #     pred_noise_uncond=pred_noise_uncond,
        #     classifier_network=cls_net,
        #     label=y
        # )

        # cg_coef = -(1 - alpha_bar_t).sqrt()

        # cg_term = cg_coef * self.lambda_cg * grad

        cfg_term = 7.5*(pred_noise_cond - pred_noise_uncond)

        pred_noise = pred_noise_cond + cfg_term

        eps = torch.tensor(EPSILON).reshape(-1, 1, 1, 1).to(self.device)
        model_mean = (1 / (alpha_t.sqrt() + eps)) * (
            x_t - ((beta_t / ((1 - alpha_bar_t).sqrt() + eps)) * pred_noise)
        )
        model_var = beta_t

        if t > 0:
            noise = self.sample_noise(batch_size=x_t.size(0))
        else:
            noise = torch.zeros_like(x_t)

        return model_mean + (model_var ** 0.5) * noise

    @torch.no_grad()
    def perform_denoising_process(self, gen_net, cls_net, x_T, y, n_frames=None):
        if n_frames is not None:
            frames = list()

        x_t = x_T
        pbar = tqdm(range(self.n_diffusion_steps - 1, -1, -1), leave=False)
        for t in pbar:
            pbar.set_description("Denoising...")

            x_t = self.p_sample(gen_net, cls_net, x_t, t=t, y=y)

            if n_frames is not None and (
                t % (self.n_diffusion_steps // n_frames) == 0
            ):
                frames.append(self._get_frame(x_t))

        return frames if n_frames is not None else x_t

    @torch.no_grad()
    def sample(self, gen_net, cls_net, batch_size, label):
        x_T = self.sample_noise(batch_size)

        return self.perform_denoising_process(
            x_T=x_T,
            gen_net=gen_net,
            cls_net=cls_net,
            y=label,
            n_frames=None,
        )