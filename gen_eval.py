import os

from torch.utils.data import ConcatDataset

from dataset import get_dataset_new_task
from dataset_utils import ImageFolderDataset
from path_manager import PathManager

from fid import get_fid_value

def eval_gen_dataset(dataset_name, class_idx_lst, manager: PathManager, device = 'cpu'):
    fids = {}

    real_dataset = ConcatDataset([
        get_dataset_new_task(dataset_name = dataset_name, class_indicies = class_indicies)
        for class_indicies in class_idx_lst
    ])

    gen_dataset = ImageFolderDataset(manager.get_image_path(aug=False))
    aug_dataset = ImageFolderDataset(manager.get_image_path(aug=True))

    fids['gen'] = get_fid_value(real_dataset, gen_dataset, device=device)

    if len(aug_dataset) > 0:
        fids['aug'] = get_fid_value(real_dataset, aug_dataset, device=device)

        combined_dataset = ConcatDataset([gen_dataset, aug_dataset])

        fids['comb'] = get_fid_value(real_dataset, combined_dataset, device=device)
        
        
    # Save accs to a text file
    fids_txt_path = os.path.join(manager.get_results_path(), 'task_fids.txt')
    with open(fids_txt_path, 'w') as f:
        for genre, fid in fids.items():
            f.write(f"{genre}: {fid}\n")
    print(f"FID values saved to {fids_txt_path}")