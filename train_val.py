from glob import glob
import os
import sys
import time
from collections import OrderedDict

import numpy as np
from sklearn.model_selection import train_test_split

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.functional as tf
import torch.utils.data as data
from torch.cuda import amp
import torch.backends.cudnn as cudnn

from tqdm import tqdm
import pickle
import pandas as pd

from metrics import *
from utils import *
from model import *
from dataset import *
from loss import *

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def train(train_loader, model, loss_fn, optimizer, device): #scaler
    losses = AverageMeter()
    ious = AverageMeter()
    dices = AverageMeter()

    model.train()                                       
    for batch_idx, (data, labels) in tqdm(enumerate(train_loader), total=len(train_loader)):    # train
        data = data.to(device)#, dtype=torch.float)
        labels = labels.to(device)#, dtype=torch.long) 

        model.zero_grad()
        # forward
        #with amp.autocast(): 
        preds = model(data)                         
        train_loss = loss_fn(preds, labels)
        train_iou_scores = iou_score(preds, labels)
        train_dice_scores = dice_coef(preds, labels)
        
        losses.update(train_loss.item(), data.size(0))
        ious.update(train_iou_scores, data.size(0))
        dices.update(train_dice_scores, data.size(0))

        # backward
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        # scaler.scale(train_loss).backward()
        # scaler.step(optimizer)
        # scaler.update()
        #torch.cuda.empty_cache()                                       

    log = OrderedDict([
    ('loss', losses.avg),
    ('iou', ious.avg),
    ('dice', dices.avg),
    ])
    
    return log



def validate(val_loader, model, loss_fn, device):
    losses = AverageMeter()
    ious = AverageMeter()
    dices = AverageMeter()
    model.eval()

    with torch.no_grad():
        for batch_idx, (data, labels) in tqdm(enumerate(val_loader), total=len(val_loader)):        # val
            data = data.to(device) # , dtype=torch.float)
            labels = labels.to(device) # , dtype=torch.long) 
            #with amp.autocast():
            
            preds = model(data)
            val_loss = loss_fn(preds, labels)
            val_iou_scores = iou_score(preds, labels)
            val_dice_scores = dice_coef(preds, labels)               
            
            losses.update(val_loss.item(), data.size(0))
            ious.update(val_iou_scores, data.size(0))
            dices.update(val_dice_scores, data.size(0))

            #torch.cuda.empty_cache()          

    log = OrderedDict([
    ('loss', losses.avg),
    ('iou', ious.avg),
    ('dice', dices.avg),
    ])

    return log
                                
def main(): # 70% 15% 15% split
    EPOCHS = 10000
    LEARNING_RATE = 1e-4
    EARLY_STOP = 20
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    BATCH_SIZE = 32
    NUM_WORKERS = 0
    #SCALER = amp.GradScaler()
    TRAIN_IMG_DIR = glob(r'D:\\AI MSc Large Modules\\Masters_Project\\CODE\\Deep-Inter-Active-Refinement-Network-for-Medical-Image-Segmentation\\data\\train_val\\img\\*')
    TRAIN_LABEL_DIR = glob(r'D:\\AI MSc Large Modules\\Masters_Project\\CODE\\Deep-Inter-Active-Refinement-Network-for-Medical-Image-Segmentation\\data\\train_val\\label\\*')

    train_img_paths, val_img_paths, train_mask_paths, val_mask_paths = train_test_split(TRAIN_IMG_DIR, TRAIN_LABEL_DIR, test_size=0.2, random_state=41)
    print("train_num:%s"%str(len(train_img_paths)))
    print("val_num:%s"%str(len(val_img_paths)))

    # model, loss criteria, optimiser
    model = UNet2D(in_channels=4, out_channels=3).to(DEVICE) 
    print("=> Creating 2D UNET Model")
    print(count_params(model))
    loss_fn = BCEDiceLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)

    # instantiate datasets
    train_dataset = Dataset(train_img_paths, train_mask_paths) 
    val_dataset = Dataset(val_img_paths, val_mask_paths)

    # instantiate dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=True,
        drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=True,
        drop_last=False)

    log = pd.DataFrame(index=[], columns=[
        'epoch', 'lr', 'loss', 'iou', 'dice', 'val_loss', 'val_iou', 'val_dice'
    ])

    best_iou = 0
    trigger = 0
    start = time.time()
    for epoch in range(EPOCHS):
        print("'Epoch [%d/%d]" %(epoch, EPOCHS))

        # train
        train_log = train(train_loader, model, loss_fn, optimizer, DEVICE)

        # validate
        val_log = validate(val_loader, model, loss_fn, DEVICE)

        print('loss %.4f - iou %.4f - dice - %.4f - val_loss %.4f - val_iou %.4f - val_dice %.4f'
            %(train_log['loss'], train_log['iou'], train_log['dice'], val_log['loss'], val_log['iou'], val_log['dice']))
        
        tmp = pd.Series([
            epoch,
            LEARNING_RATE,
            train_log['loss'],
            train_log['iou'],
            train_log['dice'],
            val_log['loss'],
            val_log['iou'],
            val_log['dice'],
        ], index=['epoch', 'lr', 'loss', 'iou', 'dice', 'val_loss', 'val_iou', 'val_dice'])

        log = log.append(tmp, ignore_index=True)
        log.to_csv('base_training_results/log.csv', index=False)

        trigger += 1

        if val_log['iou'] > best_iou:
            torch.save(model.state_dict(), 'models/2DUNET.pth')
            best_iou = val_log['iou']
            print("=> best model has been saved")
            trigger = 0

        # early stopping
        if not EARLY_STOP is None:
            if trigger >= EARLY_STOP:
                print("=> early stopping")
                break
        
        # if (epoch % 5==0):
        #     checkpoint = {
        #         "state_dict": model.state_dict(), 
        #         "optimizer": optimizer.state_dict(),}
        #     save_checkpoint(checkpoint, 'UNET_training.pth.tar')

        torch.cuda.empty_cache()

    end = time.time()
    print('Training and validation has taken ', (end - start)/60, 'minutes to complete')

if __name__ == '__main__':
    main()