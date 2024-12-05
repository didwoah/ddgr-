import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import Subset, DataLoader, ConcatDataset, Dataset
import random
import os

from dataset_utils import RelabeledDataset, ImageFolderDataset, save_as_image

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

def get_generate_dataset(folder_path, generator, total_size, batch_size, label_pool, device) -> ImageFolderDataset:

    file_index = 0

    batch_schedule = [batch_size] * (total_size//batch_size) + [total_size%batch_size]

    # 균등하게 라벨 scheduling
    label_schedule = []
    while len(label_schedule) < total_size:
        for label in label_pool:
            label_schedule.append(label)

            if len(label_schedule) == total_size:
                break
    
    random.shuffle(label_schedule)

    # batch schedule에 맞게 label schedule 자르기
    label_batches = []
    start = 0
    for batch in batch_schedule:
        label_batches.append(label_schedule[start:start + batch])
        start += batch

    # 샘플링 - {파일 번호}_{라벨}.png 로 folder path에 저장
    for batch, labels in zip(batch_schedule, label_batches):
        labels = torch.tensor(labels).to(device)
            
        images = generator.sample(batch, labels)

        for image, label in zip(images, labels):
            file_name = f"{file_index}_{label}.png"
            save_as_image(image, os.path.join(folder_path, file_name))

            file_index += 1

    return ImageFolderDataset(folder_path=folder_path)

def get_dataset_config(dataset_name):
    assert dataset_name in DATASET_CONFIGS.keys()
    
    return DATASET_CONFIGS[dataset_name]
    
DATASET_CONFIGS = {
    'mnist': {'size': 32, 'channels': 1, 'classes': 10},
    'mnist-color': {'size': 32, 'channels': 3, 'classes': 10},
    'cifar10': {'size': 32, 'channels': 3, 'classes': 10},
    'cifar100': {'size': 32, 'channels': 3, 'classes': 100},
    'svhn': {'size': 32, 'channels': 3, 'classes': 10},
}