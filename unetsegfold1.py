import os

os.environ["CUDA_VISIBLE_DEVICES"] = '2,3'
os.environ["OMP_NUM_THREADS"] = '32'
from functools import wraps
from pathlib import Path
import cv2
import IPython
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from cv2 import INTER_NEAREST
import random
from PIL import Image
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle
from sympy import Ne, interpolate
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter
from torchvision import models
from torchvision import transforms
from torchvision import transforms as T
from tqdm import tqdm
import copy
from byol_pytorch import BYOL
from dice_scorecopy import dice_loss
from evaluatecopy import evaluate, trainevaluate, u2plevaluate,u2plevaluateval
from modules import *
# from unetmodel import UNet
from ablationnet.segmentation.Models import AttU_Net,NestedUNet
from unet import UNet
print(os.environ["CUDA_VISIBLE_DEVICES"])
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from ablationnet.unetmodels.cenet import CE_Net_,CE_Net_OCT,CE_Net_backbone_DAC_with_inception
from sklearn.model_selection import KFold
from torch.utils.data import SubsetRandomSampler
# from teacher import Net,CAM
# default `log_dir` is "runs" - we'll be more specific here
# writer = SummaryWriter('/home/wjc20/segmentation/byol/newidea/unet/log/retrain/conexp/newcutoutgridmask') #
writer = SummaryWriter('/home/segmentation/byol/newidea/unet/review/log/table1new/gridmask/1')
dir_checkpoint = Path('/home/segmentation/byol/newidea/unet/review/pth/table1new/gridmask/1')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
BATCH_SIZE = 24
moving_average_decay = 0.99
EPOCHS     = 100
LR         = 1e-3
NUM_GPUS   = 2
IMAGE_SIZE = 256
IMAGE_EXTS = ['.jpg', '.png', '.jpeg']
import logging
from os import listdir
from os.path import splitext
from pathlib import Path

from sklearn.utils.multiclass import type_of_target
from bdfset import UnlabelData,MyData,TestData,TestData1,Copypastedata,Cutoutdata,gridmaskdata,Labeldata
def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2
def mixup_data(x, y, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    # if alpha > 0:
    lam = np.random.beta(alpha, alpha)
    # else:
        # lam = 1

    batch_size = x.size()[0]
    
    index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def train_net(net1,
              device,
              epochs=EPOCHS,
              batch_size = BATCH_SIZE,
              learning_rate = LR,
              val_percent: float = 0.8,
              save_checkpoint: bool = True,
              amp: bool = False):
    # dataset = MyData()
    dataset = gridmaskdata()
    #Split into train / validation partitions
    train_number = int(len(dataset) * val_percent)
    # val_number = int((len(dataset) - train_number))
    tnumber = int(0.1 * (len(dataset)- train_number))
    val_number = int(len(dataset) - train_number - tnumber)
    train_set , val_set ,tset = random_split(dataset, [train_number, val_number,tnumber], generator=torch.Generator().manual_seed(3407))
    
    
    kf = KFold(n_splits=5, shuffle=True,random_state=3407)
    aa = kf.split(train_set)
    for fold, (train_idx, val_idx) in enumerate(aa):
        
        if fold ==1:
            training_idx = train_idx
            valu_idx = val_idx
    train_sampler = SubsetRandomSampler(training_idx) 
    val_sampler = SubsetRandomSampler(valu_idx)
    train_loader = DataLoader(train_set, sampler=train_sampler,shuffle=False, batch_size=BATCH_SIZE,drop_last=True)
    val_loader = DataLoader(train_set, sampler= val_sampler,shuffle=False, batch_size=1,drop_last=True)
    optimizer = optim.RMSprop(net1.parameters(), lr=LR, weight_decay=1e-5, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor = 0.2, patience=6)  # goal: maximize Dice score  factor = 0.5,!!!!!!
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    #criterion = nn.CrossEntropyLoss(ignore_index=255)
    global_step = 0
    testset = TestData()
    testset1 = TestData1()
    
    testloader1 = DataLoader(testset1,shuffle=True,batch_size=1,drop_last=True)
    
    for epoch in range(epochs):
        net1.train()
        
        with tqdm(total=train_number*0.8, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                
                
                # uv = batch[1]
                labels = batch[1]//200
                labels  = labels.to(device=device, dtype=torch.long)
                images = batch[0]
                images = images.to(device=device, dtype=torch.float32)
                # images = images/255
                
                # uv = uv.to(device=device, dtype=torch.float32)
                # uv = uv.to(device=device, dtype=torch.float32)
                criterion = nn.CrossEntropyLoss(weight = torch.from_numpy(np.array([0.2,0.8])).float(),ignore_index=2).cuda(device)
                #pred = F.one_hot(labels_pred.argmax(dim=1), 21).permute(0, 3, 1, 2).float()
                images = torch.squeeze(images)

                with torch.cuda.amp.autocast(enabled=amp):
                    
                    labels_pred = net1(images)
                    # IPython.embed()
                    
                    labels = torch.squeeze(labels)
                
                    # labels2 = torch.squeeze(labels2)
                    onehotpred = F.softmax(labels_pred,dim=1).float()
                    loss1 = criterion(labels_pred, labels) 
                    
                    labeldice = copy.deepcopy(labels)
                    labeldice[labeldice==2]=0
                    labeldice  = labeldice.to(device=device, dtype=torch.long)
                    
                    labeldice = F.one_hot(labeldice, 2).permute(0, 3 ,1, 2).float()  
                    # onehotpred = F.one_hot(onehotpred.argmax(dim=1),2).permute(0,3,1,2).float()
                    loss2 = dice_loss(onehotpred[:,:,...],
                                     labeldice[:, :, ...],
                                       multiclass=False)
                    loss = loss1 + loss2
                    # IPython.embed() 
                    # loss = loss3 + loss4
                    if global_step % 30 == 0:
                        writer.add_scalar('train crossloss',loss1.item(),global_step=global_step)
                        writer.add_scalar('train diceloss',loss2.item(),global_step=global_step)

                optimizer.zero_grad()
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
      
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # Evaluation round
                division_step = (train_number*0.8 // (1 * batch_size))
                               # Evaluation round
                if division_step > 0:
                  if epoch > 10:
                    # if epoch % 2== 0:
                    if global_step % division_step == 0:
                                # division_step = (train_number // (1 * batch_size))
                            val_score, val_score2,val_score3,val_score4,val_score5= u2plevaluateval(net1, val_loader, device)   
                        
                            logging.info('Validation Dice score: {}'.format(val_score))
                            logging.info('Validation IOU score: {}'.format(val_score2))
                            print('learning rate:',optimizer.param_groups[0]['lr'],)                  
                            print('epoch:',epoch)
                            # if global_step % 10 == 0:
                            writer.add_scalar('validation Dice score',val_score, global_step=global_step)
                            writer.add_scalar('validation IOU score',val_score2, global_step=global_step)
                            writer.add_scalar('validation Precision',val_score3, global_step=global_step)
                            writer.add_scalar('validation Recall',val_score4, global_step=global_step)
                            writer.add_scalar('validation Accuracy',val_score5, global_step=global_step)

       
                    if global_step % division_step == 0:
                            val_score, val_score2,val_score3,val_score4,val_score5 = u2plevaluate(net1, testloader1, device)
                          
                            
                            scheduler.step(val_score)
                            # print(len(testloader))

                            logging.info('Test Dice score: {}'.format(val_score))
                            logging.info('Test IOU score: {}'.format(val_score2))
                            print('learning rate:',optimizer.param_groups[0]['lr'],)
                        
                            print('epoch:',epoch)
                           
                          
                            # if global_step % 10 == 0:
                            writer.add_scalar('test247 Dice score',val_score, global_step=global_step)
                            writer.add_scalar('test247 IOU score',val_score2, global_step=global_step)
                            writer.add_scalar('test247 Precision',val_score3, global_step=global_step)
                            writer.add_scalar('test247 Recall',val_score4, global_step=global_step)
                            writer.add_scalar('test247 Acc',val_score5, global_step=global_step)

        if save_checkpoint:
          if epoch > 20:
           if epoch%10 == 0:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)

            torch.save({
            'epoch': epoch,
            'model_state_dict': net1.module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch + 1)))
            logging.info(f'Checkpoint {epoch + 1} saved!')

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
# 设置随机数种子


if __name__ == '__main__':
    setup_seed(3407)
 
    backbone = UNet(out_channels=2)
    pthfile = r'/home/segmentation/unetbackbone3.pth' 
    pretext = torch.load(pthfile,map_location=device)

    pretext.outc = OutConv(64, 2)
    backbone = pretext
    # backbone.load_state_dict(pretext_model['model_state_dict'],strict=False)
    # IPython.embed()
    # backbone.load_state_dict(pretext['model_state_dict'],strict=False)    
    backbone = torch.nn.DataParallel(backbone)
    backbone = backbone.cuda()


    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logging.info(f'Using device {device}')
    
    try:
        train_net(net1=backbone,
                #   net2=backbone2,
                  epochs=EPOCHS,
                  batch_size=BATCH_SIZE,
                  learning_rate=LR,
                  device=device,
                  val_percent=0.97
                  )
    except KeyboardInterrupt:
        torch.save(backbone.module.state_dict(), '/home/wjc20/segmentation/pth/u2pl/epoch.pth')
        logging.info('Saved interrupt')

