import os
import json
from torch.utils.data import Dataset

import torch
from torchvision.transforms import ToPILImage, ToTensor
from PIL import Image

def _get_map(curr_classes, saver):

    start = 0

    prev_map = saver.get_map()

    if len(prev_map) > 0:
        start = max(prev_map.values()) + 1

    new_labels = [label for label in curr_classes if label not in prev_map.keys()]
    new_map = {label: idx + start for idx, label in enumerate(new_labels)}
    updated_map  = {**prev_map, **new_map}

    saver.set_map(updated_map)

    return updated_map 

class RelabeledDataset(Dataset):
    def __init__(self, dataset, curr_classes, saver, test=False):
        self.dataset = dataset
        if not test:
            self.label_map = _get_map(curr_classes, saver)
        else:
            self.label_map = saver.get_map()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data, original_label = self.dataset[idx]
        if type(original_label) == int:
            new_label = self.label_map[original_label]
            return data, torch.tensor(original_label), torch.tensor(new_label)
        else:
            new_label = self.label_map[original_label.item()]
        return data, torch.tensor(original_label), torch.tensor(new_label)
    
class EmptyDataset(Dataset):
    def __len__(self):
        return 0
    def __getitem__(self, idx):
        raise IndexError("This dataset is empty.")
    

def files_in_directory(directory_path):
    all_items = os.listdir(directory_path)
    files = [f for f in all_items if os.path.isfile(os.path.join(directory_path, f))]
    return files

def tensorToPIL(tensor: torch.Tensor) -> Image:
    tensor = tensor.clone()
    tensor = tensor.div(2).add(0.5)

    return ToPILImage()(tensor)

def save_as_image(tensor, file_path):
    image = tensorToPIL(tensor)
    image.save(file_path)
    
class ImageFolderDataset(Dataset):
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.image_paths = files_in_directory(folder_path)

        self.transform = ToTensor()

    def __len__(self):
         return len(self.image_paths)
    
    def get_label_from_image_path(self, image_path: str):
        name, extensoin = os.path.splitext(os.path.basename(image_path))
        return int(name.split('_')[-1])
    
    def __getitem__(self, index):
        img_path = os.path.join(self.folder_path, self.image_paths[index])
        image = self.transform(Image.open(img_path))
        label = torch.tensor(self.get_label_from_image_path(img_path), dtype=torch.long)
        
        return image, label