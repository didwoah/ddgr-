import torch
import torch.nn as nn
from tqdm import tqdm
import argparse
import os
import json

from dataset import split_task, get_dataset_new_task, get_loader, get_dataset_config, get_generate_dataset
from utils import get_diffusion_model
from classifier.AlexNet.AlexNet import get_new_task_classifier
from cls_train import task_classifier_train, eval_classifier

from diffusion.id12.method import Id12Method
from diffusion.id12.model import Id12
from diffusion.id12.network import UNet
from saver import Saver

from copy import deepcopy
import diffusion
import logger as Logger

def main(args, saver : Saver):

    logger = Logger.FileLogger("./save/experiments0", "log.txt")
    logger.on()

    if args.dataset == 'cifar100':
        class_idx_lst = split_task(args.class_nums, 100)
    elif args.dataset == 'cifar10':
        class_idx_lst = split_task(args.class_nums, 10)

    dataset_config = get_dataset_config(args.dataset)
    n_classes = sum(args.class_nums)

    diffusion_model = get_diffusion_model(dataset_config, args, class_idx_lst)      # id1, id2, id12 중에 택하는 함수
    
    # diffusion_method = Id12Method(
    #             img_size=dataset_config['size'],
    #             image_channels=dataset_config['channels'],
    #             device=args.device,
    #             lambda_cg=args.lambda_cg,
    #             lambda_cfg=args.lambda_cfg,)
    
    # diffusion_model = Id12(
    #             diffusion_network=None,
    #             prev_diffusion_network=None,
    #             diffusion_method=diffusion_method,
    #             diffusion_optimizer=None,
    #             classifier_network=None,
    #             label_pool=class_idx_lst[0],
    #             prev_label_pool=None,
    #             distillation_ratio=args.distillation_ratio,
    #             distillation_label_ratio=args.distillation_label_ratio,
    #             distillation_trial=args.distillation_trial
    #         )

    for task in range(len(class_idx_lst)):
        print(f'task {task} start~')

        new_task_dataset, _ = get_dataset_new_task(args.dataset, class_idx_lst[task])

        if task == 0 and args.pre_train:
            # setting
            loader = get_loader([new_task_dataset], args.cls_batch_size, class_idx_lst[task], saver)

            # classifier pretraininig
            classifier = get_new_task_classifier(sum(args.class_nums[:task+1]), prev_model_path = None, head_shared = args.head_shared, device = args.device)
            cls_optimzier = torch.optim.Adam(classifier.parameters(), lr=args.cls_lr)
            classifier = task_classifier_train(classifier, loader, cls_optimzier, epochs=args.cls_epochs, device = args.device)

            # classifier save
            save_path = os.path.join(saver.get_model_path('classifier'), 'weights.pth')
            torch.save(classifier.state_dict(), save_path)

            # generator pretraining
            
            diffusion_network = UNet(
                n_classes=sum(args.class_nums[:task+1]),
                img_channels=dataset_config['channels']
            ).to(args.device)

            diffusion_optimizer = torch.optim.Adam(diffusion_network.parameters(), lr=args.gen_lr)

            diffusion_model.diffusion_network = diffusion_network
            diffusion_model.diffusion_optimizer = diffusion_optimizer

            loader = get_loader([new_task_dataset], args.gen_batch_size, class_idx_lst[task], saver)
            diffusion_model.train(loader, iters=args.gen_iters, device=args.device)

            # diffusion network saving for distillation
            if args.distillation:
                diffusion_model.prev_diffusion_network = deepcopy(diffusion_model.diffusion_network)
                diffusion_model.prev_diffusion_network.eval()
                diffusion_model.prev_label_pool = diffusion_model.label_pool
                
            # generator save
            save_path = os.path.join(saver.get_model_path('generator'), 'weights.pth')
            torch.save(diffusion_network.state_dict(), save_path)

            saver.update_task_count()
            continue

        # get generated image dataset
        prev_classes = []
        for i in range(task):
            prev_classes.extend(class_idx_lst[i])
        
        diffusion_network = UNet(
            n_classes=len(prev_classes),
            img_channels=dataset_config['channels']
        ).to(args.device)
        prev_model_path = os.path.join(saver.get_prev_model_path('generator'), 'weights.pth')
        diffusion_network.load_state_dict(torch.load(prev_model_path))

        # Extend label embedder
        diffusion_network.extend_embedder(n_classes=n_classes, prev_n_classes=len(prev_classes), device=args.device)
        diffusion_model.label_pool = diffusion_model.label_pool + len(class_idx_lst[task])

        diffusion_optimizer = torch.optim.Adam(diffusion_network.parameters(), lr=args.gen_lr)

        prev_model_path = os.path.join(saver.get_prev_model_path('classifier'), 'weights.pth')
        classifier = get_new_task_classifier(
            sum(args.class_nums[:task+1]), 
            sum(args.class_nums[:task]), 
            prev_model_path, args.head_shared,
            device=args.device)

        diffusion_model.diffusion_network = diffusion_network
        diffusion_model.diffusion_optimizer = diffusion_optimizer
        diffusion_model.classifier_network = classifier

        generated_dataset = get_generate_dataset(
            folder_path=saver.get_image_path(),
            generator=diffusion_model,
            total_size=int(len(new_task_dataset)*args.generate_ratio),
            batch_size=args.gen_batch_size,
            label_pool=prev_classes,
            saver=saver,
            device=args.device) #
        

        # classifier train
        cls_loader = get_loader([new_task_dataset, generated_dataset], args.cls_batch_size, class_idx_lst[task], saver)

        prev_model_path = os.path.join(saver.get_prev_model_path('classifier'), 'weights.pth')
        classifier = get_new_task_classifier(sum(args.class_nums[:task+1]), sum(args.class_nums[:task]), prev_model_path, args.head_shared)

        cls_optimzier = torch.optim.Adam(classifier.parameters(), lr=args.cls_lr)
        classifier = task_classifier_train(classifier, cls_loader, cls_optimzier, epochs=args.cls_epochs, device = args.device)

        # eval acc & classifier save
        eval_classifier(classifier, args.dataset, class_idx_lst[:task + 1], saver, device = args.device)

        save_path = os.path.join(saver.get_model_path('classifier'), 'weights.pth')
        torch.save(classifier.state_dict(), save_path)

        # diffusion distillation checking
        if args.distillation:
            gen_loader = get_loader([new_task_dataset], args.gen_batch_size, class_idx_lst[task], saver)
        else:
            gen_loader = get_loader([new_task_dataset, generated_dataset], args.gen_batch_size, class_idx_lst[task], saver)
        
        # diffusion train
        diffusion_model.train(gen_loader, iters=args.gen_iters, device=args.device)

        # diffusion network saving for distillation
        if args.distillation:
            diffusion_model.prev_diffusion_network = deepcopy(diffusion_model.diffusion_network)
            diffusion_model.prev_diffusion_network.eval()
            diffusion_model.prev_label_pool = diffusion_model.label_pool
            
        save_path = os.path.join(saver.get_model_path('generator'), 'weights.pth')
        torch.save(diffusion_network.state_dict(), save_path)

        # total task evaluation


        # eval fid
        saver.update_task_count()

    logger.off()



def arg():
    parser = argparse.ArgumentParser(description="Example of argparse usage")

    # 명령줄 인자 추가
    parser.add_argument("--dataset", type=str, default="cifar100", choices=["cifar100", "cifar10"], help="dataset name")
    parser.add_argument("--cls_model", type=str, default="alexnet", choices=["alexnet", "resnet"], help="Choose classifier")

    parser.add_argument("--method", type=str, default="id12", choices=["id1", "id2", "id12"], help="Modelling methods")
    
    parser.add_argument("--distillation", action="store_true", help="Using distillation or not")
    parser.add_argument("--distillation_ratio", type=float, default=0.5, help="Distillation weight")
    parser.add_argument("--distillation_label_ratio", type=float, default=1.0, help="Number of labels for distillation")
    parser.add_argument("--distillation_trial", type=int, default=1, help="Number of sampling per each label")

    parser.add_argument("--cls_batch_size", type=int, default=32, help="Batch size for training classifier")
    parser.add_argument("--gen_batch_size", type=int, default=8, help="Batch size for training generator")

    parser.add_argument("--cls_epochs", type=int, default=50, help="Number of epochs of classifier")
    
    parser.add_argument("--gen_iters", type=int, default=50000, help="Number of iterations of generator")


    parser.add_argument("--cls_lr", type=float, default=1e-4, help="Learning rate of classifier")
    parser.add_argument("--gen_lr", type=float, default=0.001, help="Learning rate of generator")


    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"], help="Device to train on")

    parser.add_argument("--generate_ratio", type=float, default=0.5, help="relative size of generated dataset")
    parser.add_argument("--lambda_cfg", type=float, default=1, help="weight of classifier free guidance score")
    parser.add_argument("--lambda_cg", type=float, default=1, help="weight of classifier guidance score")

    parser.add_argument("--class_nums", type=list, default=[20, 20, 20, 20, 20,], help="[50, 5, 5, 5, 5,]")

    parser.add_argument("--map_path", type=str, default="./", help="class map path")
    parser.add_argument("--head_shared", action="store_true", help="Whether to share classifier head across tasks")
    parser.add_argument("--pre_train", action="store_true", help="Whether to perform pre-training on task 0")


    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = arg()
    saver = Saver(args)
    main(args, saver)