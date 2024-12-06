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


def label_squeezing_collate_fn(batch):
    # for b in batch:
    #     print(b)
    x, y = default_collate(batch)
    return x, y.long().squeeze()


def get_data_loader(dataset, batch_size, cuda=False, collate_fn=None, drop_last=False):
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        collate_fn=(collate_fn or default_collate),
        drop_last=drop_last,
        **({'num_workers': 0, 'pin_memory': True} if cuda else {})
    )


def save_checkpoint(model, model_dir):
    path = os.path.join(model_dir, model.name)

    # save the checkpoint.
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    torch.save({'state': model.state_dict()}, path)

    # notify that we successfully saved the checkpoint.
    print('=> saved the model {name} to {path}'.format(
        name=model.name, path=path
    ))


def load_checkpoint(model, model_dir):
    path = os.path.join(model_dir, model.name)

    # load the checkpoint.
    checkpoint = torch.load(path)
    print('=> loaded checkpoint of {name} from {path}'.format(
        name=model.name, path=path
    ))

    # load parameters and return the checkpoint's epoch and precision.
    model.load_state_dict(checkpoint['state'])


def test_model(model, sample_size, path, verbose=True):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torchvision.utils.save_image(
        model.sample(sample_size).data,
        path + '.jpg',
        nrow=6,
    )
    if verbose:
        print('=> generated sample images at "{}".'.format(path))


def validate(model, dataset, test_size=1024,
             cuda=False, verbose=True, collate_fn=None):
    data_loader = get_data_loader(
        dataset, 128, cuda=cuda,
        collate_fn=(collate_fn or default_collate),
    )
    total_tested = 0
    total_correct = 0
    for data, labels in data_loader:
        # break on test size.
        if total_tested >= test_size:
            break
        # test the model.
        data = Variable(data).cuda() if cuda else Variable(data)
        labels = Variable(labels).cuda() if cuda else Variable(labels)
        scores = model(data)
        _, predicted = torch.max(scores, 1)

        # update statistics.
        total_correct += (predicted == labels).sum().data.item()
        total_tested += len(data)

    precision = total_correct / total_tested
    if verbose:
        print('=> precision: {:.3f}'.format(precision))
    return precision


def xavier_initialize(model):
    modules = [m for n, m in model.named_modules() if 'conv' in n or 'fc' in n]
    parameters = [p for m in modules for p in m.parameters()]

    for p in parameters:
        if p.dim() >= 2:
            nn.init.xavier_normal(p)
        else:
            nn.init.constant(p, 0)


def gaussian_intiailize(model, std=.01):
    modules = [m for n, m in model.named_modules() if 'conv' in n or 'fc' in n]
    parameters = [p for m in modules for p in m.parameters()]

    for p in parameters:
        if p.dim() >= 2:
            nn.init.normal_(p, std=std)
        else:
            nn.init.constant_(p, 0)

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

class LambdaModule(nn.Module):
    def __init__(self, f):
        super().__init__()
        self.f = f

    def forward(self, x):
        return self.f(x)
