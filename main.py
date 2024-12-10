import torch
import torch.nn as nn
from tqdm import tqdm
import argparse
import os
import json

from dataset import split_task, get_dataset_new_task, get_loader, get_dataset_config, get_generated_dataset
from classifier.AlexNet.AlexNet import get_new_task_classifier
from cls_train import task_classifier_train, eval_classifier

from diffusion.proposed.network import UNet
from diffusion.proposed.cfg import CFGModule
from diffusion.proposed.dual_sample import DualGuidedModule
from diffusion.proposed.diffusion_kd import DiffusionKDModule
from diffusion.proposed.trainer import DiffusionTrainer
from diffusion.proposed.var_scheduler import DDPMScheduler, DDIMScheduler
from diffusion.proposed.lr_scheduler import GradualWarmupScheduler

from path_manager import PathManager

from copy import deepcopy
import logger as Logger

from gen_eval import eval_gen_dataset

def main(args, manager : PathManager):

    gen_iters = ( 50000 * args.gen_epochs ) // args.gen_batch_size
    logger = Logger.FileLogger(manager.get_results_path(), "log.txt")
    logger.on()

    if args.dataset == 'cifar100':
        class_idx_lst = split_task(args.class_nums, 100)
        num_labels = 100
    elif args.dataset == 'cifar10':
        class_idx_lst = split_task(args.class_nums, 10)
        num_labels = 10
    
    for task in range(len(class_idx_lst)):
        print(f'task {task} start~')
        
        # gen iters define
        gen_iters = ( 100 * sum(class_idx_lst[:task+1]) * args.gen_epochs ) // args.gen_batch_size

        new_task_dataset, _ = get_dataset_new_task(args.dataset, class_idx_lst[task])

        if task == 0 and args.pre_train:
            # setting
            dataloader = get_loader([new_task_dataset], args.cls_batch_size, class_idx_lst[task], manager)

            # classifier pretraininig
            cls_network = get_new_task_classifier(sum(args.class_nums[:task+1]), prev_model_path = None, head_shared = args.head_shared, device = args.device)
            cls_optimzier = torch.optim.AdamW(cls_network.parameters(), lr=args.cls_lr)
            cls_network = task_classifier_train(cls_network, dataloader, cls_optimzier, epochs=args.cls_epochs, device = args.device)

            # classifier save
            save_path = manager.get_model_path('classifier')
            torch.save(cls_network.state_dict(), save_path)

            # generator pretraining
            gen_network = UNet(T=args.T, num_prev_labels=sum([0]+args.class_nums[:task]), num_labels=num_labels, ch=args.channel, ch_mult=args.channel_mult,
                     num_res_blocks=args.num_res_blocks, dropout=args.dropout).to(args.device)

            gen_optimizer = torch.optim.AdamW(gen_network.parameters(), lr=args.gen_lr, weight_decay=1e-4)
            cosineScheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=gen_optimizer, T_max=args.gen_epochs, eta_min=0, last_epoch=-1)

            warmUpScheduler = GradualWarmupScheduler(optimizer=gen_optimizer, multiplier=args.multiplier, warm_epoch=args.gen_epochs // 10, after_scheduler=cosineScheduler)

            var_scheduler = DDPMScheduler(args.T, args.beta_1, args.beta_T, args.device) if not args.ddim else DDIMScheduler(args.T, args.beta_1, args.beta_T, args.ddim_sampling_steps, args.eta, args.device)
            cfg_model = CFGModule(gen_network, var_scheduler, args.ddim, args.cfg_factor, args.device)

            trainer = DiffusionTrainer(cfg_model, manager)

            trainer.train(dataloader, gen_optimizer, gen_iters, warmUpScheduler)
                
            # generator save
            save_path = manager.get_model_path('generator')
            cfg_model.save(save_path)

            manager.update_task_count()
            logger.off()
            logger = Logger.FileLogger(manager.get_results_path(), "log.txt")
            logger.on()
            continue

        # init netwrok
        prev_cls_model_path = manager.get_prev_model_path('classifier')
        cls_network = get_new_task_classifier(sum(args.class_nums[:task+1]), sum(args.class_nums[:task]), prev_cls_model_path, args.head_shared, device=args.device)

        gen_network = UNet(T=args.T, num_prev_labels=sum([0]+args.class_nums[:task]), num_labels=100, ch=args.channel, ch_mult=args.channel_mult,
                     num_res_blocks=args.num_res_blocks, dropout=args.dropout).to(args.device)
        gen_network.load_state_dict(torch.load(manager.get_prev_model_path('generator')))
        
        # get generated image dataset
        var_scheduler = DDPMScheduler(args.T, args.beta_1, args.beta_T, args.device) if not args.ddim else DDIMScheduler(args.T, args.beta_1, args.beta_T, args.ddim_sampling_steps, args.eta, args.device)
        cfg_model = CFGModule(gen_network, var_scheduler, args.ddim, args.cfg_factor, args.device)

        num_gen_samples = int(args.gen_ratio * (sum(args.class_nums[:task]) * 500))

        generated_dataset = get_generated_dataset(
            cfg_model, 
            num_gen_samples,
            args.gen_batch_size, 
            sum(args.class_nums[:task]), 
            manager, 
            args.device)
        
        augmented_dataset = None
        
        if args.dual_guidance:
            num_aug_samples = int(args.aug_ratio * (sum(args.class_nums[:task]) * 500))

            dual_guided_model = DualGuidedModule(
                gen_network, 
                var_scheduler,
                cls_network,
                args.ddim,
                args.cg_factor,
                args.cfg_factor,
                args.device)
            augmented_dataset = get_generated_dataset(
                dual_guided_model, 
                num_aug_samples, 
                args.gen_batch_size, 
                sum(args.class_nums[:task]), 
                manager, 
                args.device,
                aug = True)
            
        # classifier train
        cls_loader = get_loader([new_task_dataset, generated_dataset, augmented_dataset], args.cls_batch_size, class_idx_lst[task], manager)
        
        print(f'{len(cls_loader)} samples are concated.')

        cls_optimzier = torch.optim.AdamW(cls_network.parameters(), lr=args.cls_lr)
        cls_network = task_classifier_train(cls_network, cls_loader, cls_optimzier, epochs=args.cls_epochs, device = args.device)

        # eval acc & classifier save
        eval_classifier(cls_network, args.dataset, class_idx_lst[:task + 1], manager, device = args.device)

        save_path = manager.get_model_path('classifier')
        torch.save(cls_network.state_dict(), save_path)

        # generator train
        gen_optimizer = torch.optim.AdamW(gen_network.parameters(), lr=args.gen_lr, weight_decay=1e-4)
        cosineScheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=gen_optimizer, T_max=args.gen_epochs, eta_min=0, last_epoch=-1)

        warmUpScheduler = GradualWarmupScheduler(optimizer=gen_optimizer, multiplier=args.multiplier, warm_epoch=args.gen_epochs // 10, after_scheduler=cosineScheduler)

        var_scheduler = DDPMScheduler(args.T, args.beta_1, args.beta_T, args.device) if not args.ddim else DDIMScheduler(args.T, args.beta_1, args.beta_T, args.ddim_sampling_steps, args.eta, args.device)
        cfg_model = CFGModule(gen_network, var_scheduler, args.ddim, args.cfg_factor, args.device)

        if args.diffusion_kd:
            teacher_network = deepcopy(gen_network)
            kd_model = DiffusionKDModule(
                gen_network, 
                var_scheduler, 
                teacher_network,
                sum(args.class_nums[:task]),
                args.kd_sampling_ratio,
                device = args.device
                )
            trainer = DiffusionTrainer(
                cfg_model,
                manager,
                kd_model,
                args.kd_factor,
                device = args.device
            )
            # gen_iters edit
            gen_iters = ( 100 * class_idx_lst[task] * args.gen_epochs ) // args.gen_batch_size
        else:
            trainer = DiffusionTrainer(cfg_model, manager)

        if args.diffusion_kd:
            diff_loader = get_loader([new_task_dataset], args.gen_batch_size, class_idx_lst[task], manager)
        else:
            diff_loader = get_loader([new_task_dataset, generated_dataset, augmented_dataset], args.gen_batch_size, class_idx_lst[task], manager)
            
        trainer.train(diff_loader, gen_optimizer, gen_iters, warmUpScheduler)
            
        # generator save
        save_path = manager.get_model_path('generator')
        cfg_model.save(save_path)

        # eval fid
        # eval_gen_dataset(args.dataset, class_idx_lst[:task + 1], manager, device = args.device)
        manager.update_task_count()
        logger.off()
        logger = Logger.FileLogger(manager.get_results_path(), "log.txt")
        logger.on()

    logger.off()

def arg():
    parser = argparse.ArgumentParser(description="Example of argparse usage")

    # 명령줄 인자 추가
    parser.add_argument("--dataset", type=str, default="cifar100", choices=["cifar100", "cifar10"], help="dataset name")
    parser.add_argument("--cls_model", type=str, default="alexnet", choices=["alexnet", "resnet"], help="Choose classifier")
    
    # Important
    parser.add_argument("--ddim", action="store_true", help="DDIM sampling enable")
    parser.add_argument("--diffusion_kd", action="store_true", help="Using distillation or not")
    parser.add_argument("--pre_train", action="store_true", help="Whether to perform pre-training on task 0")
    parser.add_argument("--dual_guidance", action="store_true", help="Using dual guidance or not")


    parser.add_argument("--cls_batch_size", type=int, default=64, help="Batch size for training classifier")
    parser.add_argument("--gen_batch_size", type=int, default=16, help="Batch size for training generator")

    parser.add_argument("--cls_epochs", type=int, default=100, help="Number of epochs of classifier")
    parser.add_argument("--gen_epochs", type=int, default=70, help="Number of iterations of generator")

    parser.add_argument("--ddim_sampling_steps", type=int, default=100, help="DDIM num sampling steps")
    parser.add_argument("--eta", type=float, default=0.0, help="DDIM variance coefficient for deterministic or probabilistic")

    parser.add_argument("--cls_lr", type=float, default=1e-4, help="Learning rate of classifier")
    parser.add_argument("--gen_lr", type=float, default=1e-4, help="Learning rate of generator")


    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"], help="Device to train on")

    parser.add_argument("--gen_ratio", type=float, default=.5, help="ratio generated dataset against prev samples num")
    parser.add_argument("--aug_ratio", type=float, default=.1, help="ratio augmented dataset against prev samples num")

    parser.add_argument("--class_nums", type=list, default=[50, 5, 5, 5, 5, 5], help="[50, 5, 5, 5, 5,]")

    parser.add_argument("--map_path", type=str, default="./", help="class map path")
    parser.add_argument("--head_shared", action="store_true", help="Whether to share classifier head across tasks")

    parser.add_argument("--T", type=int, default=500, help="Number of timesteps")
    parser.add_argument("--channel", type=int, default=128, help="Base number of channels")
    parser.add_argument("--channel_mult", type=int, nargs='+', default=[1, 2, 2, 2], help="Channel multiplier for each level")
    parser.add_argument("--num_res_blocks", type=int, default=2, help="Number of residual blocks per level")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--multiplier", type=float, default=2.5, help="Multiplier for diffusion loss")
    parser.add_argument("--beta_1", type=float, default=1e-4, help="Beta start value for scheduler")
    parser.add_argument("--beta_T", type=float, default=0.028, help="Beta end value for scheduler")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping value")

    parser.add_argument("--cfg_factor", type=float, default=1.8, help="Weight for classifier-free guidance")    # CFG original paper best factor : 1.8
    parser.add_argument("--cg_factor", type=float, default=-0.3, help="Weight for classifier guidance")         # CG original paper best factor : 0.3
    parser.add_argument("--kd_sampling_ratio", type=float, default=0.2)
    parser.add_argument("--kd_factor", type=float, default=5.0)                                                 # LWF original paper best factor : ?

    args = parser.parse_args()
    
    return args

if __name__ == "__main__":
    args = arg()
    saver = PathManager(args)
    main(args, saver)