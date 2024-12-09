import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion.proposed.cfg import extract, CFGModule

class DualGuidedModule(CFGModule):
    def __init__(self, network, var_scheduler, classifier_network, ddim=False, cg_factor=-1.8, cfg_factor = 1.8, device = 'cuda'):
        super().__init__(network, var_scheduler, ddim, cfg_factor, device)
        self.classifier_network = classifier_network
        self.cg_factor = cg_factor

    @torch.enable_grad()
    def get_classifier_score(self, x_t, t, eps_no_cond, y):

        x = x_t.detach().requires_grad_(True)
        # using tweedie's formula
        x_0 = (x - (1 - extract(self.var_scheduler.alphas_cumprod, t)).sqrt() * eps_no_cond) / extract(self.var_scheduler.alphas_cumprod, t).sqrt()
        
        out = self.classifier_network(x_0)

        log_prob = F.log_softmax(out, dim=-1)
        selected = log_prob[torch.arange(log_prob.size(0), dtype=torch.long), y.long()]

        return torch.autograd.grad(outputs=selected.sum(), inputs=x)[0]
 
    
    @torch.no_grad()
    def p_sample(self, x_t, prev_t, t, y):

        eps_factor = (1 - extract(self.var_scheduler.alphas, t)) / (
            1 - extract(self.var_scheduler.alphas_cumprod, t)
        ).sqrt()

        noise = torch.randn_like(x_t)

        var = (
            (1 - extract(self.var_scheduler.alphas_cumprod, prev_t)) / (1 - extract(self.var_scheduler.alphas_cumprod, t)) * extract(self.var_scheduler.betas, t)
        )
        t_expanded = t[:, None, None, None]
        noise_factor = var.sqrt()
        noise_factor = torch.where(t_expanded>1, noise_factor, torch.zeros_like(noise_factor))
        
        eps_cond = self.network(x_t, t, y)
        # edit required
        no_cond = torch.fill(y, 100).to(self.device)
        eps_no_cond = self.network(x_t, t, no_cond)
        cfg_eps = (1. + self.cfg_factor) * eps_cond - self.cfg_factor * eps_no_cond

        classifier_score = self.get_classifier_score(x_t, t, eps_no_cond, y)

        score_term = self.cg_factor * var * classifier_score

        mean = (x_t - eps_factor * cfg_eps) / extract(self.var_scheduler.alphas, t).sqrt() + score_term

        x_t_prev = mean + noise_factor * noise

        return x_t_prev
    
    @torch.no_grad()
    def p_sample_ddim(self, x_t, prev_t, t, y):

        alpha = extract(self.var_scheduler.alphas_cumprod, t)
        prev_alpha = extract(self.var_scheduler.alphas_cumprod, prev_t)

        noise = torch.randn_like(x_t).to(self.device)
    
        var = self.var_scheduler.eta * ((1 - prev_alpha) / (1 - alpha) * (1 - alpha) / prev_alpha)
        
        t_expanded = t[:, None, None, None]
        
        var = torch.where(t_expanded>1, var, torch.zeros_like(var))
        noise_factor = var.sqrt()
        
        eps_cond = self.network(x_t, t, y)
        # edit required
        no_cond = torch.fill(y, 100).to(self.device)
        eps_no_cond = self.network(x_t, t, no_cond)
        
        cfg_eps = (1. + self.cfg_factor) * eps_cond - self.cfg_factor * eps_no_cond

        classifier_score = self.get_classifier_score(x_t, t, eps_no_cond, y)

        score_term = self.cg_factor * var * classifier_score
        
        mean = prev_alpha.sqrt() / alpha.sqrt() * (x_t - ((1 - alpha).sqrt()*cfg_eps)) + (1 - prev_alpha - var).sqrt() * cfg_eps + score_term
        x_t_prev = mean + noise_factor * noise

        return x_t_prev
