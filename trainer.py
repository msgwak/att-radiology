import os
import shutil
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from dataset import YSDataset, load_dataset
from model import AttDiagnosisModel
from loss import IoU_loss

torch.set_default_tensor_type('torch.FloatTensor')

            
def train(args, repeat, trn_loader, val_loader, device):
    torch.cuda.empty_cache()
    model = AttDiagnosisModel(num_classes=args.num_classes, model_type=args.model_type)
    if len(opt_gpus) > 1:
        model = torch.nn.DataParallel(model)
    model.to(device)

    # optimizer
    if args.optim == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,  betas=(0.9, 0.999))
    elif args.optim == 'radam':
        optimizer = torch.optim.RAdam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    elif args.optim == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=1e-5, momentum=0.9)
        
    # scheduler
    if args.scheduler == 'multisteplr':
        milestones = [int(lr_drop * args.num_epoch) for lr_drop in (args.lr_drops or [])]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    elif args.scheduler == 'cosineannealing':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epoch/2)

    # criterion
    criterion_cl = nn.CrossEntropyLoss()
    criterion_att = IoU_loss
    weight_sum = args.weight_cl + args.weight_att
    
    # training
    best_acc, best_epoch, no_improve_counter = 0, 0, 0       
    for epoch in range(args.num_epoch): 
        train_total, train_correct, train_loss = 0, 0, 0
        if isinstance(model, torch.nn.DataParallel):
            model.module.model.training = True
        else:
            model.model.training = True
        for i, (images, attlabels, labels, flags) in enumerate(tqdm(trn_loader, ncols=100, ascii=True, leave=False)):
            images, attlabels, labels, flags = images.to(device), attlabels.to(device), labels.to(device), flags.to(device)
            logits, attmasks, _ = model(images, labels)
            loss_cl = criterion_cl(logits, labels)
            loss_att = criterion_att(attmasks, attlabels, flags)
            loss = ((args.weight_cl*loss_cl) + (args.weight_att*loss_att)) / weight_sum
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_total += images.size(0)
            _, predicted = F.softmax(logits, dim=1).max(1)
            train_correct += predicted.eq(labels).sum().item()
        scheduler.step()
        train_loss = train_loss / train_total
        train_acc = train_correct / train_total * 100

        ################ Validation ################
        val_total, val_correct, val_loss = 0, 0, 0
        predicted_list, labels_list = [], []
        if isinstance(model, torch.nn.DataParallel):
            model.module.model.training = False
        else:
            model.model.training = False
        with torch.no_grad():
            for i, (images, attlabels, labels, flags) in enumerate(val_loader):
                images, attlabels, labels, flags = images.to(device), attlabels.to(device), labels.to(device), flags.to(device)
                logits, attmasks, _ = model(images, labels)
                loss_cl = criterion_cl(logits, labels)
                loss_att = criterion_att(attmasks, attlabels, flags)
                loss = ((args.weight_cl*loss_cl) + (args.weight_att*loss_att)) / weight_sum
                val_loss += loss.item()
                val_total += images.size(0)
                _, predicted = F.softmax(logits,dim=1).max(1)
                predicted_list.extend(predicted.cpu().numpy())
                labels_list.extend(labels.cpu().numpy())
        val_correct = np.equal(np.array(predicted_list), np.array(labels_list)).sum()
        val_loss = val_loss / val_total
        val_acc = val_correct / val_total * 100

        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            no_improve_counter = 0
        else:
            no_improve_counter += 1
            if no_improve_counter >= args.patience:
                print(f'>> No improvement for {args.patience} epochs. Stop at epoch {epoch}!')
                break
        ################################################

        print(f'>> Epoch [{epoch}/{args.num_epoch}] \t Train Loss: {train_loss:.5f} \t Val Loss: {val_loss:.5f} \t Train Acc: {train_acc:.2f} \t Val Acc: {val_acc:.2f}')
        print(f'>> best acc: {best_acc:.4f} at epoch {best_epoch}  (no improve: {no_improve_counter} / {args.patience})')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # runner arg
    parser.add_argument("--seed", type=int, default=42)
    
    # data arg
    parser.add_argument("--num_classes", type=int, default=3)
    parser.add_argument("--RAW_IMG_W", type=int, default=1280)
    parser.add_argument("--RAW_IMG_H", type=int, default=720)
    parser.add_argument("--IMG_SIZE_W", type=int, default=940)
    parser.add_argument("--IMG_SIZE_H", type=int, default=520)
    parser.add_argument("--clahe", action='store_true', default=False)
    parser.add_argument("--rotate", action='store_true', default=False)
    parser.add_argument("--intensity", action='store_true', default=False)
    parser.add_argument("--flip", action='store_true', default=False)
    parser.add_argument("--resize", action='store_true', default=False)
    parser.add_argument("--jawratio", action='store_true', default=False)
    parser.add_argument("--max_resize_rate", type=float, default=0.03)
    parser.add_argument("--max_jaw_ratio_rate", type=float, default=0.05)
    parser.add_argument("--label_ratio", type=int, default=100)

    # training arg
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--model_type", type=str, default='resnet50')
    parser.add_argument("--optim", type=str, default='SGD')
    parser.add_argument("--scheduler", type=str, default='cosineannealing')
    parser.add_argument("--num_epoch", type=int, default=100)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--lr_drops", type=float, nargs=3, default=[0.2, 0.4, 0.75])
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_cl", type=float, default=2)
    parser.add_argument("--weight_att", type=float, default=1)
    args = parser.parse_args()

    # SEED
    np.random.seed(args.seed)
    
    # DEVICE
    opt_gpus = [i for i in range(torch.cuda.device_count())]
    if len(opt_gpus) > 1:
        print("Using ", len(opt_gpus), " GPUs")
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(x) for x in opt_gpus)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # DATA / TRANSFORMS
    transform_list, mask_transform_list = [], []
    load_data = load_dataset
    transform_list.append(transforms.CenterCrop(size=(args.IMG_SIZE_H,args.IMG_SIZE_W)))
    mask_transform_list.append(transforms.CenterCrop(size=(args.IMG_SIZE_H,args.IMG_SIZE_W)))
    if args.intensity:
        transform_list.append(transforms.ColorJitter(brightness=.05, contrast=.1))
    transform_list.append(transforms.ToTensor())
    mask_transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    mask_transform = transforms.Compose(mask_transform_list)
    data_dict, masks_dict, targets_dict, flags_dict = load_data(args) # Load custom data here
    
    # DATASET / DATALOADER
    trn_dataset = YSDataset(data_dict['train'], masks_dict['train'], targets_dict['train'], flags_dict['train'], args, simul_random_aug=True, transform=transform, mask_transform=mask_transform)
    val_dataset = YSDataset(data_dict['val'], masks_dict['val'], targets_dict['val'], flags_dict['val'], args, simul_random_aug=False, transform=mask_transform, mask_transform=mask_transform)
    trn_loader = DataLoader(dataset=trn_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    if "train" in args.mode:
        train(args, repeat, trn_loader, val_loader, device)
