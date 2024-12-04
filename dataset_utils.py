import json
from torch.utils.data import Dataset

def _get_map(labels, map_path):

    start = 0

    with open(map_path, "r") as file:
        prev_map = json.load(file)

    if not isinstance(prev_map, dict):
        prev_map = {}

    if len(prev_map) > 0:
        start = max(prev_map.values()) + 1

    new_labels = [label for label in labels if label not in prev_map]
    new_map = {label: idx + start for idx, label in enumerate(new_labels)}
    updated_map  = prev_map | new_map

    with open(map_path, "w") as file:
        json.dump(updated_map, file, indent=4)

    return updated_map 

class RelabeledDataset(Dataset):
    def __init__(self, dataset, map_path):
        self.dataset = dataset
        original_labels = sorted(set(label for _, label in dataset))
        self.label_map = _get_map(original_labels, map_path)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data, original_label = self.dataset[idx]
        new_label = self.label_map[original_label]
        return data, original_label, new_label