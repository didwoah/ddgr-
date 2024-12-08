import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion.proposed.cfg import extract, CFGModule

class DualGuidedModule(CFGModule):
    def __init__(self, network, var_scheduler, classifier_network, cg_factor, cfg_factor = 1.8, device = 'cuda'):
        super().__init__(network, var_scheduler, cfg_factor, device)
        self.classifier_network = classifier_network
        self.cg_factor = cg_factor

    @torch.enable_grad()
    def get_classifier_score(self, x_t, y):

        x = x_t.detach().requires_grad_(True)
        out = self.classifier(x)

        log_prob = F.log_softmax(out, dim=-1)
        selected = log_prob[torch.arange(log_prob.size(0), dtype=torch.long), y.long()]

        return torch.autograd.grad(outputs=selected.sum(), inputs=x_t)[0]
    
    @torch.no_gard()
    def p_sample(self, x_t, t, y):

        if isinstance(t, int):
            t = torch.tensor([t]).to(self.device)

        eps_factor = (1 - extract(self.scheduler.alphas, t)) / (
            1 - extract(self.scheduler.alphas_cumprod, t)
        ).sqrt()

        noise = torch.randn_like(x_t)

        var = (
            (1 - extract(self.scheduler.alphas_cumprod, t-1)) / (1 - extract(self.scheduler.alphas_cumprod, t)) * extract(self.scheduler.betas, t)
        )
        t_expanded = t[:, None]
        var = torch.where(t_expanded>1, var, torch.zeros_like(var))
        noise_factor = var.sqrt()

        eps_no_cond = self.network(x_t, t, None)
        eps_cond = self.network(x_t, t, y)
        cfg_eps = (1. + self.cfg_factor) * eps_cond - self.cfg_factor * eps_no_cond

        classifier_score = self.get_classifier_score(x_t, y)
        score_term = self.cg_factor * var * classifier_score

        mean = (x_t - eps_factor * cfg_eps) / extract(self.scheduler.alphas, t).sqrt() +  score_term

        x_t_prev = mean + noise_factor * noise

        return x_t_prev
