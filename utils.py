import os
import os.path
import random

import torchvision
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ToPILImage, ToTensor

from PIL import Image


def get_diffusion_model(dataset_config, args, class_idx_lst):
    from diffusion import id1, id2, id12
    
    if args.method == "id1":
        pass
    elif args.method == "id2":
        diffusion_method = id2.Id2Method(
                img_size=dataset_config['size'],
                image_channels=dataset_config['channels'],
                device=args.device,
                lambda_cg=args.lambda_cg,
                lambda_cfg=args.lambda_cfg,)
    
        diffusion_model = id2.Id2(
                    diffusion_network=None,
                    prev_diffusion_network=None,
                    diffusion_method=diffusion_method,
                    diffusion_optimizer=None,
                    classifier_network=None,
                    label_pool=len(class_idx_lst[0]),
                    prev_label_pool=None,
                    distillation_ratio=args.distillation_ratio,
                    distillation_label_ratio=args.distillation_label_ratio,
                    distillation_trial=args.distillation_trial
                )
    elif args.method == "id12":
        diffusion_method = id12.Id12Method(
                img_size=dataset_config['size'],
                image_channels=dataset_config['channels'],
                device=args.device,
                lambda_cg=args.lambda_cg,
                lambda_cfg=args.lambda_cfg,)
    
        diffusion_model = id12.Id12(
                    diffusion_network=None,
                    prev_diffusion_network=None,
                    diffusion_method=diffusion_method,
                    diffusion_optimizer=None,
                    classifier_network=None,
                    label_pool=len(class_idx_lst[0]),
                    prev_label_pool=None,
                    distillation_ratio=args.distillation_ratio,
                    distillation_label_ratio=args.distillation_label_ratio,
                    distillation_trial=args.distillation_trial
                )
    
    return diffusion_model       


def sample(nums, count):
    selected = random.sample([i for i in range(nums)], count)
    # for i in selected:
    #     list.remove(i)
    return selected

def tensorToPIL(tensor: torch.Tensor) -> Image:
    tensor = tensor.clone()
    tensor_min = tensor.min()
    tensor_max = tensor.max()
    tensor = (tensor - tensor_min) / (tensor_max - tensor_min)  # 정규화

    return ToPILImage()(tensor)

def save_as_image(tensor, file_path):
    image = tensorToPIL(tensor)
    image.save(file_path)

def files_in_directory(directory_path):
    # 폴더 내 모든 파일과 디렉토리 가져오기
    all_items = os.listdir(directory_path)
    # 파일만 필터링
    files = [f for f in all_items if os.path.isfile(os.path.join(directory_path, f))]
    return files
