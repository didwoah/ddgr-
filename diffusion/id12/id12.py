import os

import random

import torch
from torch import nn
import torch.nn.functional as F

from .classifier_guidance import ClassifierGuidedDiffusion

from utils import sample

class Id12(nn.Module):
    def __init__(self, 
                 model: ClassifierGuidedDiffusion,
                 n_classes,
                 lambda_cg,
                 lambda_cfg,
                 cg_cfg_ratio,
                 distillation_ratio,
                 distillation_label_ratio,
                 distillation_trial):
        super().__init__()

        self.device = next(model.parameters()).device
        self.model = model
        self.n_classes = n_classes

        self.lambda_cg = lambda_cg
        self.lambda_cfg = lambda_cfg
        self.cg_cfg_ratio = cg_cfg_ratio

        self.distillation_ratio = distillation_ratio
        self.distillation_label_ratio = distillation_label_ratio
        self.distillation_trial = distillation_trial
        
        self.optimizer = None
        self.classifier_optimizer = None
        self.prev_scholar = None


    def __sample(self, lambda_cg, lambda_cfg, y, batch_size):
        assert batch_size > 0

        # if len(self.label_pool) == 0:
        #     label = None
        # else:
        #     label = torch.tensor(
        #         random.choices(self.label_pool, k=batch_size)
        #         ).to(self.model.device)

        return self.model.sample(
            batch_size=batch_size,
            label=y,
            lambda_cg=lambda_cg,
            lambda_cfg=lambda_cfg)

    def sample(self, size, y=None):
        return self.__sample(self.lambda_cg, self.lambda_cfg, y, size)
        # cg_size = int(self.cg_cfg_ratio * size)
        # cfg_size = size - cg_size

        # mode_cg = self.__sample(
        #     lambda_cg=self.lambda_cg,
        #     lambda_cfg=self.lambda_cfg,
        #     batch_size=cg_size)
        
        # path = os.path.join('.', 'sample','test_2','cg')
        # os.makedirs(path, exist_ok=True)
        # for i, image in enumerate(mode_cg):
        #     save_as_image(image, os.path.join(path, f'{i}.png'))
        
        # mode_cfg = self.__sample(
        #     lambda_cg=self.lambda_cg,
        #     lambda_cfg=self.lambda_cfg,
        #     batch_size=cfg_size)
        
        # path = os.path.join('.', 'sample','test_2','cfg')
        # os.makedirs(path, exist_ok=True)
        # for i, image in enumerate(mode_cfg):
        #     save_as_image(image, os.path.join(path, f'{i}.png'))
        
        # concat = torch.concat([mode_cg, mode_cfg], dim=0)

        # return concat
    
    def classifier_noise_train_a_batch(self, x, y) -> torch.Tensor:
        rand_diffusion_step = self.model.sample_diffusion_step(batch_size=x.size(0))
        rand_noise = self.model.sample_noise(batch_size=x.size(0))
        noisy_image = self.model.perform_diffusion_process(
            ori_image=x,
            diffusion_step=rand_diffusion_step,
            rand_noise=rand_noise,
        )

        self.classifier_optimizer.zero_grad()

        loss = self.model.classifier.get_loss(noisy_image, rand_diffusion_step, y)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.model.classifier.parameters(), max_norm=1.0)

        self.classifier_optimizer.step()

        return loss


    def control_distillation_ratio(self, x, y):
        label_pool = self.label_pool[:]

        n_selection = max(int(len(label_pool)*self.distillation_label_ratio), 1)

        selected_y, _ = sample(label_pool, n_selection)

        mask = torch.tensor([False] * y.shape[0]).to(self.device)
        for selected in selected_y:
            mask = mask | (y == selected)
        if not mask.any():
            mask[0] = True
            
        selected_x = x[mask]

        return selected_x, selected_y


    def distillate(self, x, y):
        selected_x, selected_y = self.control_distillation_ratio(x, y)

        prev_model = self.prev_scholar.generator.model
        prev_model.eval()
        with torch.no_grad():
            prev_errors = self.get_classify_errors(selected_x, selected_y, prev_model)
            prev_dist = self.get_distribution_of_class(selected_y, prev_errors)

        errors = self.get_classify_errors(selected_x, selected_y, self.model)
        dist = self.get_distribution_of_class(selected_y, errors)
        
        loss = F.cross_entropy(dist, prev_dist)
        
        return loss


    def train_a_batch(self, x, y, x_=None, y_=None, importance_of_new_task=.5):
        assert x_ is None or x.size() == x_.size()
        assert y_ is None or y.size() == y_.size()

        noise_train_loss = self.classifier_noise_train_a_batch(x, y)

        self.optimizer.zero_grad()
               
        new_task_loss = self.model.get_unet_loss(x, y)
        if x_ is not None and y_ is not None:
            old_task_loss = self.model.get_unet_loss(x_, y_)

            loss = (
                importance_of_new_task * new_task_loss +
                (1-importance_of_new_task) * old_task_loss
            )
        else:
            loss = new_task_loss

        if self.prev_scholar is not None:
            ditillation_loss = self.distillate(x, y)

            alpha = self.distillation_ratio
            loss = (1-alpha)*loss + alpha*ditillation_loss

        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        self.optimizer.step()

        return {'c_loss': noise_train_loss.item(), 'g_loss': loss.item()}
    

    def classify(self, x):
        errors = self.get_classify_errors(x)

        dist = self.get_distribution_of_class(errors)

        pred, _ = min(enumerate(dist), key=lambda x: x[1])

        return pred
    
    def get_distribution_of_class(self, label_pool, errors):
        mean_errors = torch.tensor([0]).to(self.device)
        for label in sorted(label_pool):
            mean = torch.mean(errors[label]).unsqueeze(0)
            mean_errors = torch.cat([mean_errors, -mean], dim=0)
        
        return F.softmax(mean_errors[1:])
    
    def get_classify_errors(self, x, label_pool, model):
        errors = {}
        trial = self.distillation_trial
        for _ in range(trial):
            rand_diffusion_step = model.sample_diffusion_step(batch_size=x.size(0))
            rand_noise = model.sample_noise(batch_size=x.size(0))
            noisy_image = model.perform_diffusion_process(
                ori_image=x,
                diffusion_step=rand_diffusion_step,
                rand_noise=rand_noise,
            )

            for label in label_pool:
                y = torch.tensor([label]*x.shape[0]).to(self.device)
                eps_th = model(noisy_image, rand_diffusion_step, y)

                error = eps_th - rand_noise

                if label in errors.keys():
                    errors[label] = torch.cat([errors[label], error], dim=0)
                else:
                    errors[label] = error
        
        return errors
    
