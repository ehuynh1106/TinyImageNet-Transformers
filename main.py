from data_aug import get_data_augmentations
from dataset import load_train_data, load_val_data
from discord_updates import DiscordWebhook
from engine import train, validate
from log import create_logger
from math import ceil
from os import get_terminal_size
from scaler import NativeScaler
from throughput import throughput
from timm.loss.cross_entropy import SoftTargetCrossEntropy
from timm.models.cait import cait_s36_384
from timm.models.deit import deit_base_distilled_patch16_384
from timm.models.swin_transformer import swin_large_patch4_window12_384
from timm.models.vision_transformer import vit_large_patch16_384

import argparse, sys, torch
import torch.nn as nn
import torch.optim as optim

def parse_args():
    parser = argparse.ArgumentParser('Vision Transformer training and evaluation script', add_help=False)
    parser.add_argument('--model', type=str, required=True, choices=['vit', 'deit', 'swin', 'cait', 'beit'],
                        help='vit: ViT-L/16, '
                             'deit: DeiT-B/16 distilled, '
                             'swin: Swin-L, '
                             'cait: CaiT-S36')

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--train', action='store_true', help='Train a model')
    group.add_argument('--evaluate',  type=str, help='Evaluate a trained model at the given file path')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')
    parser.add_argument('--resume', type=str, help='Resume training')

    # data augmentation
    parser.add_argument('--mixup', action='store_true', default=True)
    parser.add_argument('--no-mixup', action='store_false', dest='mixup', help='Disable mixup')
    parser.add_argument('--cutmix', action='store_true', default=True)
    parser.add_argument('--no-cutmix', action='store_false', dest='cutmix', help='Disable cutmix')
    parser.add_argument('--randerase', action='store_true', default=True)
    parser.add_argument('--no-randerase', action='store_false', dest='randerase', help='Disable random erasing')
    parser.add_argument('--randaug', action='store_true', default=True)
    parser.add_argument('--no-randaug', action='store_false', dest='randaug', help='Disable rand augment')

    #optimizer
    parser.add_argument('--optim', type=str, default='adamw', choices=['adamw', 'sgd'])
    parser.add_argument('--nesterov', action='store_true', default=True, help='Use nesterov momentum for SGD')
    parser.add_argument('--no-nesterov', action='store_false', dest='nesterov', help='Disable nesterov')

    parser.add_argument('--label-smooth', type=float, default=0.1, help='Label smoothing percent')

    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate for optimizer')
    parser.add_argument('--wd', type=float, default=0.05, help='Weight decay for optimizer')

    args, _ = parser.parse_known_args()

    return args

def load_model(model_name, evaluate):
    if model_name == 'cait':
        model = cait_s36_384(pretrained=True, drop_path_rate=0.1)
        batch_size = 128
    elif model_name == 'deit':
        model = deit_base_distilled_patch16_384(pretrained=True, drop_path_rate=0.1)
        batch_size = 64
    elif model_name == 'swin':
        model = swin_large_patch4_window12_384(pretrained=True, drop_path_rate=0.1)
        batch_size = 32
    elif model_name == 'vit':
        model = vit_large_patch16_384(pretrained=True, drop_path_rate=0.1)
        batch_size = 64
    else:
        logger.error('Invalid model name, please use either cait, deit, swin, or vit')
        sys.exit(1)

    for param in model.parameters():
        param.requires_grad = False
    model.reset_classifier(num_classes=200)

    if evaluate:
        checkpoint = torch.load(evaluate)
        model.load_state_dict(checkpoint['model_state_dict'])
        del checkpoint

    return model, batch_size

def load_optimizer(args, model):
    if args.optim == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    elif args.optim == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd, momentum=0.9, nesterov=args.nesterov)
    else:
        logger.error('Invalid optimizer name, please use either adamw or sgd')
        sys.exit(1)

    return optimizer
"""
Experiments:
Train models with new experimental setting
"""
if __name__ == '__main__':
    logger = create_logger()
    args = parse_args()

    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, batch_size = load_model(args.model, args.evaluate)
    true_batch_size = 128
    update_freq = true_batch_size // batch_size
    img_size = 384

    logger.info(f"Parameters: {sum(p.numel() for p in model.parameters())}")

    val_loader = load_val_data(img_size, batch_size if not args.throughput else 32)
    
    if args.throughput:
        logger.info(f"Testing throughput of {args.model}")
        model = model.to(device)
        throughput(val_loader, model, logger)
    elif args.evaluate:
        logger.info(f"Evaluating {args.model} on validation set")
        model = model.to(device)
        val_loss, top_1_val_acc, top_5_val_acc = validate(model, device, val_loader, -1)
        logger.info(f"Top 1 Validation Accuracy: {top_1_val_acc}\tTop 5 Validation Accuracy: {top_5_val_acc}")
    elif args.train:
        randaug_magnitude = 9 if args.randaug else 0
        start_epoch = 0
        epochs = 30

        train_loader = load_train_data(img_size, randaug_magnitude, batch_size)
        
        optimizer = load_optimizer(args, model)

        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=ceil(len(train_loader)/update_freq)*epochs)
        loss_scaler = NativeScaler()

        mixup, random_erase = get_data_augmentations(
            args.label_smooth,
            en_mixup=args.mixup,
            en_cutmix=args.cutmix,
            en_randerase=args.randerase
        )
    
        if mixup:
            loss_fn = SoftTargetCrossEntropy()
        else:
            loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smooth)
            
        if args.resume:
            file_path = args.resume
            model_name = file_path.split('/')[-1]
            start_epoch = int(model_name[5:model_name.find('.tar')])
            checkpoint = torch.load(file_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(device)

            optimizer = load_optimizer(args, model)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader)*epochs)

            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            loss_scaler.load_state_dict(checkpoint['loss_scaler_state_dict'])
            logger.info(f"Resuming training {args.model} at epoch {start_epoch + 1}")
            logger.info(f"Current lr: {optimizer.param_groups[0]['lr']}")
            del checkpoint
        else:
            logger.info(f"Start training {args.model}")
            model = model.to(device)

        for i in range(start_epoch, epochs):
            logger.info(f"Epoch {i+1}")
            train_loss = train(model, loss_fn, optimizer, device, train_loader, scheduler, loss_scaler, update_freq, mixup, random_erase)
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss_scaler_state_dict': loss_scaler.state_dict()
            }, f'models/epoch{i+1}.tar')
            val_loss, top_1_val_acc, top_5_val_acc = validate(model, device, val_loader, i, can_visualize=i>=epochs//2)
            logger.info(f"Current lr: {optimizer.param_groups[0]['lr']}")
            logger.info(f"Top 1 Validation Accuracy: {top_1_val_acc}\tTop 5 Validation Accuracy: {top_5_val_acc}")
            print("-"*get_terminal_size().columns)

