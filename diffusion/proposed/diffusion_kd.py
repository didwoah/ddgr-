import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion.proposed.cfg import CFGModule

class DiffusionKDModule(CFGModule):
    def __init__(self, network, var_scheduler, teacher_network, num_prev_labels, kd_sampling_ratio, cfg_factor = 1.8, device = 'cuda'):
        super().__init__(network, var_scheduler, cfg_factor, device)
        self.teacher_network = teacher_network
        self.num_prev_labels = num_prev_labels
        self.kd_sampling_ratio = kd_sampling_ratio

        self.num_sampled_labels = int(num_prev_labels * kd_sampling_ratio)
    
    def loss_function(self, x_0):

        batch_size = x_0.shape[0]
        t = (
            torch.randint(1, self.var_scheduler.num_steps, size=(batch_size,))
            .to(self.device)
            .long()
        )

        noise = torch.randn_like(x_0)
        x_t = self.q_sample(x_0, t, noise)

        expanded_noise = noise.unsqueeze(1).repeat(1, self.num_sampled_labels, 1, 1, 1).view(-1, noise.shape[1], noise.shape[2], noise.shape[3])
        expanded_x_t = x_t.unsqueeze(1).repeat(1, self.num_sampled_labels, 1, 1, 1).view(-1, x_t.shape[1], x_t.shape[2], x_t.shape[3])
        expanded_t = t.unsqueeze(1).repeat(1, self.num_sampled_labels).reshape(-1,)
        
        y = torch.randint(0, self.num_prev_labels, (batch_size * self.num_sampled_labels, ), device=self.device)

        eps_student = self.network(expanded_x_t, expanded_t, y)
        with torch.no_grad():
            eps_teacher = self.teacher_network(expanded_x_t, expanded_t, y)

        l2_student = torch.norm(eps_student - expanded_noise, p=2, dim=(1, 2, 3), keepdim=True).view(batch_size, self.num_sampled_labels, 1)
        l2_teacher = torch.norm(eps_teacher - expanded_noise, p=2, dim=(1, 2, 3), keepdim=True).view(batch_size, self.num_sampled_labels, 1)

        student_softmax = F.softmax(-l2_student, dim=2)
        teacher_softmax = F.softmax(-l2_teacher, dim=2)
        loss = F.kl_div(student_softmax.log(), teacher_softmax, reduction='batchmean')

        return loss







    
