import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import Subset, DataLoader, ConcatDataset, Dataset
import random
import os
from tqdm import tqdm

from dataset_utils import RelabeledDataset, ImageFolderDataset, save_as_image, EmptyDataset
from diffusion.proposed.cfg import CFGModule

TRANSFORM = transforms.Compose([
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
            transform=TRANSFORM
        )
        test_dataset = datasets.CIFAR100(
            root="./data",
            train=False,
            download=True,
            transform=TRANSFORM
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


def get_loader(datasets, batch_size, curr_classes, saver, shuffle=True, num_workers=0, test=False):

    if not isinstance(datasets, list) or len(datasets) == 0:
        raise ValueError
    
    datasets = [ds if ds is not None else EmptyDataset() for ds in datasets]
    
    combined_dataset = ConcatDataset(datasets)
    dataset = RelabeledDataset(combined_dataset, curr_classes, saver, test)
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )

    return loader

def get_generate_dataset(generator, total_size, batch_size, label_pool, manager, device) -> ImageFolderDataset:

    folder_path = manager.get_image_path()
    file_index = 0

    map = manager.get_map()

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
        org_labels = labels[:]
        labels = [map[label] for label in labels]
        labels = torch.tensor(labels).to(device)
            
        images = generator.sample(batch, labels)

        for image, label in zip(images, org_labels):
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


def get_generated_dataset(
        generator: CFGModule, 
        num_samples, 
        batch_size, 
        num_prev_labels, 
        manager, 
        device) -> ImageFolderDataset:

    folder_path = manager.get_image_path()
    file_index = 0

    map = manager.get_map()
    inverse_map = {value: key for key, value in map.items()}

    labels = torch.randint(0, num_prev_labels, (num_samples,), device=device)
    
    org_labels = [inverse_map[label.item()] for label in labels]

    num_batches = (num_samples + batch_size - 1) // batch_size

    for i in tqdm(range(num_batches), desc="Generating Batches", unit="batch"):
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, num_samples)
        batch_labels = labels[start_idx:end_idx]

        shape = ((end_idx - start_idx), 3, 32, 32)
        images = generator.sample(shape, batch_labels)

        for image, label in zip(images, org_labels):
            file_name = f"{file_index}_{label}.png"
            save_as_image(image, os.path.join(folder_path, file_name))

            file_index += 1

    return ImageFolderDataset(folder_path=folder_path)