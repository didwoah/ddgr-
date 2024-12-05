import torch
import torch.nn as nn
from tqdm import tqdm
import argparse
import os
import json

from dataset import split_task, get_dataset_new_task, get_loader
from classifier.AlexNet.AlexNet import get_new_task_classifier
from cls_train import task_classifier_train, eval_classifier
from saver import Saver


def main(args, saver : Saver):

    if args.dataset == 'cifar100':
        class_idx_lst = split_task(args.class_nums, 100)
    elif args.dataset == 'cifar10':
        class_idx_lst = split_task(args.class_nums, 10)

    for task in range(len(class_idx_lst)):
        print(f'task {task} start~')

        new_task_dataset, _ = get_dataset_new_task(args.dataset, class_idx_lst[task])

        if task == 0 and args.pre_train:
            # setting
            loader = get_loader([new_task_dataset], args.cls_batch_size, saver)

            # classifier pretraininig
            classifier = get_new_task_classifier(sum(args.class_nums[:task+1]), prev_model_path = None, head_shared = args.head_shared, device = args.device)
            cls_optimzier = torch.optim.Adam(classifier.parameters(), lr=args.cls_lr)
            classifier = task_classifier_train(classifier, loader, cls_optimzier, device = args.device)

            # classifier save
            save_path = os.path.join(saver.get_model_path('classifier'), 'weights.pth')
            torch.save(classifier.state_dict(), save_path)

            # generator pretraining

            # generator save

            saver.update_task_count()
            continue

        # get generated image dataset
        prev_classes = []
        for i in range(task):
            prev_classes.extend(class_idx_lst[i])
        
        generated_dataset, _ = get_dataset_new_task(args.dataset, class_idx_lst[task-1])

        # classifier train
        cls_loader = get_loader([new_task_dataset, generated_dataset], args.cls_batch_size, saver)

        prev_model_path = os.path.join(saver.get_prev_model_path('classifier'), 'weights.pth')
        classifier = get_new_task_classifier(sum(args.class_nums[:task+1]), sum(args.class_nums[:task]), prev_model_path, args.head_shared)

        cls_optimzier = torch.optim.Adam(classifier.parameters(), lr=args.cls_lr)
        classifier = task_classifier_train(classifier, cls_loader, cls_optimzier, device = args.device)

        # eval acc & classifier save
        eval_classifier(classifier, args.dataset, class_idx_lst[:task + 1], saver, device = args.device)

        save_path = os.path.join(saver.get_model_path('classifier'), 'weights.pth')
        torch.save(classifier.state_dict(), save_path)

        # diffusion train


        # total task evaluation


        # eval fid
        saver.update_task_count()




def arg():
    parser = argparse.ArgumentParser(description="Example of argparse usage")

    # 명령줄 인자 추가
    parser.add_argument("--dataset", type=str, default="cifar100", choices=["cifar100", "cifar10"], help="dataset name")
    parser.add_argument("--cls_model", type=str, default="alexnet", choices=["alexnet", "resnet"], help="Choose classifier")


    parser.add_argument("--cls_batch_size", type=int, default=32, help="Batch size for training classifier")
    parser.add_argument("--gen_batch_size", type=int, default=32, help="Batch size for training generator")

    parser.add_argument("--cls_epochs", type=int, default=10, help="Number of epochs of classifier")
    parser.add_argument("--gen_epochs", type=int, default=10, help="Number of epochs of generator")


    parser.add_argument("--cls_lr", type=float, default=1e-4, help="Learning rate of classifier")
    parser.add_argument("--gen_lr", type=float, default=0.001, help="Learning rate of generator")


    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"], help="Device to train on")

    parser.add_argument("--lambda_cfg", type=float, default=0.001, help="weight of classifier free guidance score")
    parser.add_argument("--lambda_cg", type=float, default=0.001, help="weight of classifier guidance score")

    parser.add_argument("--class_nums", type=list, default=[50, 5, 5, 5, 5,], help="[50, 5, 5, 5, 5,]")

    parser.add_argument("--map_path", type=str, default="./", help="class map path")
    parser.add_argument("--head_shared", action="store_true", help="Whether to share classifier head across tasks")
    parser.add_argument("--pre_train", action="store_true", help="Whether to perform pre-training on task 0")


    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = arg()
    saver = Saver(args)
    main(args, saver)