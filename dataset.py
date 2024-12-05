import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import Subset, DataLoader, ConcatDataset, Dataset
import random
import os

from dataset_utils import RelabeledDataset

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def get_task_dataset(dataset, class_indicies):
    indices  = [i for i, (_, label) in enumerate(dataset) if label in class_indicies]
    return Subset(dataset, indices)

def split_task(class_nums: list, total_class_num=100):
    assert sum(class_nums) <= total_class_num

    random_lst = list(range(total_class_num))
    random.shuffle(random_lst)
    
    out = []
    remaining = random_lst
    for num in class_nums:
        out.append(remaining[:num])
        remaining = remaining[num:]
    return out

def get_dataset(dataset_name):
    if dataset_name == 'cifar100':
        train_dataset = datasets.CIFAR100(
            root="./data",
            train=True,
            download=True,
            transform=transforms.ToTensor()
        )
        test_dataset = datasets.CIFAR100(
            root="./data",
            train=False,
            download=True,
            transform=transforms.ToTensor()
        )

    elif dataset_name == 'imagenet':
        pass
    else:
        pass
    return train_dataset, test_dataset

def get_dataset_new_task(dataset_name, class_indicies):

    assert dataset_name in ['cifar100', 'imagenet', 'cifar10']

    train_dataset, test_dataset = get_dataset(dataset_name)

    return get_task_dataset(train_dataset, class_indicies), get_task_dataset(test_dataset, class_indicies)


def get_loader(datasets, batch_size, saver, shuffle=True, num_workers=0):

    if not isinstance(datasets, list) or not all(isinstance(ds, Dataset) for ds in datasets):
        raise ValueError
    if len(datasets) == 0:
        raise ValueError
    
    combined_dataset = ConcatDataset(datasets)
    dataset = RelabeledDataset(combined_dataset, saver)
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )

    return loader