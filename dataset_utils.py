import json
from torch.utils.data import Dataset

def _get_map(labels, saver):

    start = 0

    prev_map = saver.get_map()

    if len(prev_map) > 0:
        start = max(prev_map.values()) + 1

    new_labels = [label for label in labels if label not in prev_map.keys()]
    new_map = {label: idx + start for idx, label in enumerate(new_labels)}
    updated_map  = {**prev_map, **new_map}

    saver.set_map(updated_map)

    return updated_map 

class RelabeledDataset(Dataset):
    def __init__(self, dataset, saver):
        self.dataset = dataset
        original_labels = sorted(set(label for _, label in dataset))
        self.label_map = _get_map(original_labels, saver)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data, original_label = self.dataset[idx]
        new_label = self.label_map[original_label]
        return data, original_label, new_label