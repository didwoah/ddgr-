import torch
import torch.nn as nn
from tqdm import tqdm
import argparse
import os

from dataset import split_task, get_dataset_new_task, get_loader
from classifier.AlexNet.AlexNet import get_new_task_classifier
from cls_train import task_classifier_train


def main(args):

    if args.dataset == 'cifar100':
        task_class_lst = split_task(args.class_nums, 100)
    elif args.dataset == 'cifar10':
        task_class_lst = split_task(args.class_nums, 10)

    prev_cls_path = None

    for task, task_class in enumerate(task_class_lst):
        print(f'task {task} start~')

        new_task_dataset, _ = get_dataset_new_task(args.dataset, task_class)
        out_dim = len(task_class_lst[:task+1])

        classifier = get_new_task_classifier(out_dim, prev_cls_path, args.head_shared)

        if task == 0 and args.pre_train:
            # setting
            loader = get_loader([new_task_dataset], args.cls_batch_size, args.map_path)

            # classifier pretraininig
            classifier = task_classifier_train(classifier, )
            cls_argsimizer = torch.argsim.Adam(classifier.parameters(), lr=args.cls_lr)
            classifier = task_classifier_train(classifier, loader, cls_argsimizer, device = args.device)

            # classifier save

            # generator pretraining

            # generator save

            continue

        # get generated image dataset
        prev_classes = []
        for i in range(task):
            prev_classes.extend(task_class_lst[i])
        
        # classifier train
        cls_loader = get_loader([new_task_dataset, generated_dataset], args.batch_size, args.map_path)
        classifier = task_classifier_train(classifier, )
        cls_argsimizer = torch.optim.Adam(classifier.parameters(), lr=args.cls_lr)
        classifier = task_classifier_train(classifier, cls_loader, cls_argsimizer, device = args.device)

        # classifier save


        # diffusion train


        # total task evaluation

        # eval acc

        # eval fid




def arg():
    parser = argparse.ArgumentParser(description="Example of argparse usage")

    # 명령줄 인자 추가
    parser.add_argument("--dataset", type=str, default="cifar100", choices=["cifar100", "cifar10"], help="dataset name")
    parser.add_argument("--cls_model", type=str, default="alexnet", choices=["alexnet", "resnet"], help="Choose classifier")


    parser.add_argument("--cls_batch_size", type=int, default=32, help="Batch size for training classifier")
    parser.add_argument("--gen_batch_size", type=int, default=32, help="Batch size for training generator")

    parser.add_argument("--cls_epochs", type=int, default=10, help="Number of epochs of classifier")
    parser.add_argument("--gen_epochs", type=int, default=10, help="Number of epochs of generator")


    parser.add_argument("--cls_lr", type=float, default=0.001, help="Learning rate of classifier")
    parser.add_argument("--gen_lr", type=float, default=0.001, help="Learning rate of generator")


    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"], help="Device to train on")

    parser.add_argument("--lambda_cfg", type=float, default=0.001, help="weight of classifier free guidance score")
    parser.add_argument("--lambda_cg", type=float, default=0.001, help="weight of classifier guidance score")


    args = parser.parse_args()

    return args

def create_root_path(args):
    """
    Create a root path based on the argparse arguments.
    """
    root_path = os.path.join(
        "./results",
        f"{args.dataset}",
        f"{args.cls_model}_cls",
        f"cls_bs{args.cls_batch_size}_gen_bs{args.gen_batch_size}",
        f"cls_ep{args.cls_epochs}_gen_ep{args.gen_epochs}",
        f"cls_lr{args.cls_lr}_gen_lr{args.gen_lr}",
        f"lambda_cfg{args.lambda_cfg}_lambda_cg{args.lambda_cg}",
        f"device_{args.device}"
    )

    # Ensure the directory exists
    os.makedirs(root_path, exist_ok=True)

    print(f"Root path created: {root_path}")
    return root_path


# 처음에 map json파일 만들어야함