from dataset import VideoDataset
from augment import Augments
from config import *
from efficientnet_lite3 import ESNAModel


import os
import sys
import re
import gc
import platform
import random

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, ReduceLROnPlateau

import einops
import timm
import glob
import cv2

from rich import print as _pprint
from rich.progress import track
import albumentations as A
from albumentations.pytorch import ToTensorV2

import wandb

import warnings
warnings.simplefilter('ignore')


def train_one_epoch(model, train_dataloader, optimizer, loss_fn, epoch, device, verbose=False):
    """
    Trains model for one epoch
    """
    model.train()
    running_loss = 0
    prog_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
    for batch, (frames, targets) in prog_bar:
        optimizer.zero_grad()
        
        frames = frames.to(device, torch.float)
        targets = targets.to(device, torch.float)
        
        #squeezed_frames=torch.squeeze(frames,dim=1)
        # Re arrange the frames in the format our model wants to recieve
        frames = einops.rearrange(frames, 'b h w f -> b f h w')

        #print(frames.shape)
        
        preds = model(frames)
        #print(f'shape {preds.shape}, {targets.shape}')
        loss = loss_fn(preds, targets)
        
        loss.backward()
        optimizer.step()
        
        loss_item = loss.item()
        running_loss += loss_item
        
        prog_bar.set_description(f"loss: {loss_item:.4f}")
        
       
        
        if verbose == True and batch % 20 == 0:
            print(f"Batch: {batch}, Loss: {loss_item}")
    
    avg_loss = running_loss / len(train_dataloader)
    wandb.log({
        "avg_train_loss":round(avg_loss,4)
    })
    
    return avg_loss

@torch.no_grad()
def valid_one_epoch(model, valid_dataloader, loss_fn, epoch, device, verbose=False):
    """
    Validates the model for one epoch
    """
    model.eval()
    running_loss = 0
    prog_bar = tqdm(enumerate(valid_dataloader), total=len(valid_dataloader))
    for batch, (frames, targets) in prog_bar:
        frames = frames.to(device, torch.float)
        targets = targets.to(device, torch.float)
        
        # Re arrange the frames in the format our model wants to recieve
        
        #squeezed_frames=torch.squeeze(frames,dim=1) # channel 지우기 
        
        frames = einops.rearrange(frames, 'b h w f -> b f h w')
        
        preds = model(frames)
        loss = loss_fn(preds, targets)
        
        loss_item = loss.item()
        running_loss += loss_item
        
        prog_bar.set_description(f"val_loss: {loss_item:.4f}")
        
       
        if verbose == True and batch % 10 == 0:
            print(f"Batch: {batch}, Loss: {loss_item}")
    
    avg_val_loss = running_loss / len(valid_dataloader)
    
    wandb.log({
        "avg_val_loss":round(avg_val_loss,4)
    })
    
    return avg_val_loss


def collate_fn(batch):
    max_frames=max(frames.shape[1] for frames, _ in batch)
    
    padded_frames=[]
    labels=[]
    
    for frames,label in batch:
        padded = np.pad(frames, ((0, 0), (0, max_frames - frames.shape[1]), (0, 0)), mode='constant', constant_values=0)
        padded_frames.append(padded)
        labels.append(label)
        
    padded_frames=np.stack(padded_frames,axis=0)
    labels=np.stack(labels,axis=0)
    #labels=torch.tensor(labels)
    return torch.from_numpy(padded_frames).float(),torch.from_numpy(labels).float()


if __name__=='__main__':
    wandb.init(project='dscapstone',name='efficientnet_lite3',reinit=True)
    if torch.cuda.is_available():
        device=torch.device('cuda')
    
    
    torch.autograd.set_detect_anomaly(True)

    # train
    train_ann=pd.read_csv('/home/irteam/junghye-dcloud-dir/dscapstone/video2frames_AllFrames_NormalIncluded/train_anno.csv')
    train_dataset=VideoDataset(train_ann,classes,Augments.train_augments,is_test=False)

    # test
    valid_ann=pd.read_csv('/home/irteam/junghye-dcloud-dir/dscapstone/video2frames_AllFrames_NormalIncluded/new_valid_anno.csv')
    valid_dataset=VideoDataset(valid_ann,classes,Augments.valid_augments,is_test=False)

    print(f'Size of Training Set : {len(train_dataset)}, Size of Validation Set : {len(valid_dataset)}')

    model=ESNAModel()
    model=model.to(device)

    wandb.watch(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=Config['LR'])

    train_loss_fn = nn.BCEWithLogitsLoss()
    valid_loss_fn = nn.BCEWithLogitsLoss()

    # data loader

    train_loader=DataLoader(
        train_dataset,
        batch_size=Config['TRAIN_BS'],
        shuffle=True,
        num_workers=Config['NUM_WORKERS'],
        collate_fn=collate_fn
    )

    valid_loader=DataLoader(
        valid_dataset,
        batch_size=Config['VALID_BS'],
        shuffle=False,
        num_workers=Config['NUM_WORKERS'],
        collate_fn=collate_fn
    )

    current_loss=1000

    for epoch in range(Config['EPOCHS']):
            print(f"\n{'--'*8} EPOCH: {epoch+1} {'--'*8}\n")
            
            train_loss = train_one_epoch(model, train_loader, optimizer, train_loss_fn, epoch=epoch, device=device)
            
            valid_loss = valid_one_epoch(model, valid_loader, valid_loss_fn, epoch=epoch, device=device)
            
            print(f"val_loss: {valid_loss:.4f}")
            
            if valid_loss < current_loss:
                current_loss = valid_loss
                torch.save(model.state_dict(), f"./models/model_{Config['FEATURE_EXTRACTOR']}_{Config['NUM_CLASSES']}_{Config['EPOCHS']}.pt")
