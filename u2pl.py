
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '2,3'
os.environ["OMP_NUM_THREADS"] = '32'
import argparse
import copy
import logging
import math
import multiprocessing
import random
import sys
import time  # /home/wjc20/segmentation/byol/newidea/unetseg.py
from collections import OrderedDict
from configparser import Interpolation
from functools import wraps
# import aug as trans
from pathlib import Path

import cv2
import IPython
import lightly.data as data
import lightly.loss as loss
import lightly.models as models
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from cv2 import INTER_NEAREST
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

import wandb
from byol_pytorch import BYOL
from dice_score import dice_loss
from evaluate import evaluate, trainevaluate,u2plevaluate,u2plevaluateval
from modules import *
from unet import UNet
from bdfset import UnlabelData,MyData,TestData,TestData1
# from teacher import Net,CAM
print(os.environ["CUDA_VISIBLE_DEVICES"])
import torch.distributed as dist
from torch.nn import SyncBatchNorm
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
sys.path.append("..") 
# from baseline.U2PL.u2pl.utils.dist_helper import setup_distributed
# default `log_dir` is "runs" - we'll be more specific here
writer = SummaryWriter('/home/wjc20/segmentation/byol/newidea/unet/log/retrain/ablation/embed') #

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#dir_checkpoint = Path('./unetpth/onlyuse2modalcrossloss')
dir_checkpoint = Path('/home/wjc20/segmentation/byol/newidea/unet/pth/ablation/embed')

# a = pred_u_teacher
# constants
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
BATCH_SIZE = 10
moving_average_decay = 0.99
EPOCHS     = 200
LR         = 1e-3
NUM_GPUS   = 2
IMAGE_SIZE = 256
IMAGE_EXTS = ['.jpg', '.png', '.jpeg']
#NUM_WORKERS = multiprocessing.cpu_count()
# pytorch lightning module
import logging
from os import listdir
from os.path import splitext
from pathlib import Path
import yaml
from sklearn.utils.multiclass import type_of_target
parser = argparse.ArgumentParser(description="Semi-Supervised Semantic Segmentation")
parser.add_argument("--config", type=str, default="config.yaml")
parser.add_argument("--local_rank", type=int, default=0)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--port", default=None, type=int)

def train_net(net1,
              net2,
              device,
              epochs=EPOCHS,
              batch_size = BATCH_SIZE,
              learning_rate = LR,
              val_percent: float = 0.4,
              save_checkpoint: bool = True,
              amp: bool = False):
    # cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)
    dataset = MyData()
    undataset = UnlabelData()
    #Split into train / validation partitions
    train_number = int(len(dataset) * val_percent)
    tnumber = int(0.9 * (len(dataset)- train_number))
    val_number = int(len(dataset) - train_number - tnumber)
    number1 = int(len(undataset) * val_percent)
    number2 = int((len(undataset) - number1))
    # vnumber = len(dataset) - train_number - val_number
    #train_number = int(len(dataset)) * low_threshold
    #val_number = train_number
    # train_set , val_set, vset = random_split(dataset, [train_number, val_number,vnumber], generator=torch.Generator().manual_seed(3407))
    train_set , val_set ,tset = random_split(dataset, [train_number, val_number,tnumber], generator=torch.Generator().manual_seed(3407))
    undataset, undataset1 = random_split(undataset, [number2, number1], generator=torch.Generator().manual_seed(3407))
    # val_set, vset = random_split(val_set, [val_number*low_threshold, val_number-val_number*low_threshold], generator=torch.Generator().manual_seed(3407))
    # Create data loaders
   
    train_loader = DataLoader(train_set,batch_size=BATCH_SIZE,drop_last=True,shuffle = True)
    val_loader = DataLoader(val_set,  batch_size=1,drop_last=True,shuffle = True)
    unloader = DataLoader(undataset, batch_size=BATCH_SIZE,drop_last=True,shuffle = True)
    #初始化logging

    optimizer = optim.RMSprop(net1.parameters(), lr=LR, weight_decay=1e-5, momentum=0.8)
    # optimizer = optim.RMSprop(net1.parameters(), lr=LR,momentum=high_threshold)
    # optimizer = optim.RMSprop(net2.parameters(), lr=LR, weight_decay=1e-5, momentum=high_threshold)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor = 0.2, patience=3)  # goal: maximize Dice score  factor = 0.5,!!!!!!
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    
    global_step = 0
    testset = TestData()
    testset1 = TestData1()

    train_number1 = int(len(train_set) * 0.1)
    val_number1 = int((len(train_set) - train_number1))
    train_set1 , val_set1 = random_split(train_set, [train_number1, val_number1], generator=torch.Generator().manual_seed(3407))
    trainloader1 = DataLoader(train_set1,shuffle=True,batch_size=1,drop_last=True)
    testloader = DataLoader(testset,shuffle=True,batch_size=1,drop_last=True)
    testloader1 = DataLoader(testset1,shuffle=True,batch_size=1,drop_last=True)
    criterion = nn.CrossEntropyLoss(weight = torch.from_numpy(np.array([0.2,0.8])).float(),ignore_index=2).cuda(device)
    criterion1 = nn.CrossEntropyLoss(weight = torch.from_numpy(np.array([0.2,0.8])).float(),ignore_index=2).cuda(device)
    
    for epoch in range(epochs):
        net1.train()
        count = 0

        high_threshold = 0.85
        low_threshold  = 0.15
        if epoch>15:
            high_threshold = 0.75
            low_threshold  = 0.25
        elif epoch>30:
            high_threshold = 0.65
            low_threshold  = 0.35
        epoch_loss = 0
        train_matrix = 0
        # train_loader.sampler.set_epoch(epoch)
        # unloader.sampler.set_epoch(epoch)
        loader_l_iter = iter(train_loader)
        loader_u_iter = iter(unloader)
        with tqdm(total=(2*len(train_loader)*BATCH_SIZE), desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
        # for batch in tqdm(range(len(train_loader))):
            for batch in range(len(train_loader)):
                # torch.cuda.empty_cache()
                try:
                    images,labels = loader_l_iter.next()
                    # IPython.embed()
                except:
                    loader_l_iter = iter(train_loader)
                    # IPython.embed()
                    images,labels = loader_l_iter.next()
                
                labels = labels//200
                try: 
                    image_u,imgu = loader_u_iter.next()
                except:
                    loader_u_iter = iter(unloader)
                    image_u,imgu = loader_u_iter.next()
                # b_size,chanel, h, w = labels.size()
                
                images = images.to(device=device, dtype=torch.float32)
                # images = images
                labels  = labels.to(device=device, dtype=torch.long)
                image_u = image_u.to(device=device, dtype=torch.float32)
                images = torch.squeeze(images)
                image_u = torch.squeeze(image_u)
                labels = torch.squeeze(labels,dim=1)
                # IPython.embed()
                
                #pred = F.one_hot(labels_pred.argmax(dim=1), 21).permute(0, 3, 1, 2).float()
                
                zero = torch.zeros_like(labels)
                one = torch.ones_like(labels)   
                two = 2*torch.ones_like(labels)
                zero = zero.to(dtype=torch.float32)     
                one  = one.to(dtype=torch.float32)
                two = two.to(dtype=torch.float32)
                
                with torch.cuda.amp.autocast(enabled=amp):
                    # net2.eval()     #197 293 294与ema注释了也就是：
                    # with torch.no_grad():
                    #     out_t = net2(image_all)
                    # for s_params in zip(net1.parameters()):
                    
                        # print(s_params)
                    # for name, parms in net1.named_parameters():	
                    #     print(parms.requires_grad)
                
                    pred_u_teacher = net2(image_u)
                    # abc = pred_u_teacher
                    # if global_step%10 == 0:
                    #  print(torch.min(pred_u_teacher),torch.max(pred_u_teacher),'step is {}'.format(global_step))
                    
                    # pred_u_teacher = F.interpolate(pred_u_teacher, (h, w), mode="bilinear", align_corners=True)
                    pred_u_teacher = F.softmax(pred_u_teacher, dim=1)
    
                    label_u = torch.where(pred_u_teacher > high_threshold, one, pred_u_teacher)
                    label_u_dice = label_u
                    # IPython.embed()
                    label_u_dice = torch.where(pred_u_teacher <= high_threshold, zero, label_u_dice)
                    label_u = torch.where(label_u < low_threshold, zero, label_u)
                    # IPython.embed()
                    # label_u_dice = label_u
                    # label_u_dice = torch.where(torch.gt(label_u_dice, low_threshold) & torch.lt(label_u_dice, high_threshold), zero, label_u_dice)

                    label_u = torch.where(torch.gt(label_u, low_threshold) & torch.lt(label_u, high_threshold), two, label_u)
                    label_u = label_u[:,1].to(dtype=torch.long)
                    label_u_dice = label_u_dice[:,1].to(dtype=torch.long)
                    # IPython.embed()
                    if torch.min(label_u_dice)<0:
                        IPython.embed()
                        count =+1
                        continue
                    
                    num_labeled = len(images)
                    image_all = torch.cat((images,image_u))
                    # labels_pred = net1(images)
                    outs = net1(image_all)
                    pred_l, pred_u = outs[:num_labeled], outs[num_labeled:]  
                    
                    labels = torch.squeeze(labels)
                    sup_loss1 = criterion1(pred_l, labels) 
                    soft_pred = F.softmax(pred_l,dim=1).float()
                    labeldice = copy.deepcopy(labels)
                    labeldice[labeldice==2]=0
                    labeldice  = labeldice.to(device=device, dtype=torch.long)
                    labeldice = F.one_hot(labeldice, 2).permute(0, 3 ,1, 2).float() 
                    
                    sup_loss2 = dice_loss(soft_pred[:,...],
                                     labeldice[:,...],
                                       multiclass=False)
                    sup_loss = sup_loss1 +  sup_loss2
                    #sup_loss.requires_grad
# torch.sum(label_u==1),torch.sum(label_u==2),torch.sum(label_u),torch.min(label_u),torch.max(label_u)  torch.sum(labels==1),torch.sum(labels)
# torch.sum(label_dice==1),torch.sum(label_dice==2),torch.sum(label_dice)  torch.sum(low_threshold<label_u&&label_u>high_threshold)
                                        #loss2
                    net2.train()
                    with torch.no_grad():
                        out_t = net2(image_all)
                    unsup_loss1 = criterion(pred_u, label_u)
                    unsoft_pred = F.softmax(pred_u,dim=1).float()
                    
                    # if global_step==4:
                    #     IPython.embed()
                    # if global_step>10:
                    #     IPython.embed()
                    # print('\n',torch.max(label_u_dice),torch.min(label_u_dice))
                    
                    label_dice = F.one_hot(label_u_dice, 2).permute(0, 3 ,1, 2).float() 

                    unsup_loss2 = dice_loss(unsoft_pred[:,...],
                                     label_dice[:, ...],
                                       multiclass=False)
                    unsup_loss = unsup_loss1 + unsup_loss2
                    # print(sup_loss1.item(),sup_loss2.item(),unsup_loss1.item(),unsup_loss2.item())
                    loss = sup_loss +  unsup_loss

                    
                    if global_step % 20 == 0:
                        writer.add_scalar('train crossloss',sup_loss1.item(),global_step=global_step)
                        writer.add_scalar('train diceloss',sup_loss2.item(),global_step=global_step)
                        writer.add_scalar('train unlabel crossloss',unsup_loss1.item(),global_step=global_step)
                        writer.add_scalar('train unlabel diceloss',unsup_loss2.item(),global_step=global_step)
                      
                optimizer.zero_grad()
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(2*images.shape[0])
                global_step += 1
                # if global_step % 10 == 0:
                with torch.no_grad():
                    iterl = global_step 
                    ema_decay = moving_average_decay #min(1-1/((iterl-len(train_loader))+1.1),moving_average_decay)
                    for t_params, s_params in zip(net2.parameters(),net1.parameters()):
                        t_params.data = ema_decay*t_params.data + (1 - ema_decay) * s_params.data
                # IPython.embed()
                pbar.set_postfix(**{'loss (batch)': loss.item()})
        
                division_step = len(train_loader)#(train_number // (1 * batch_size))
                # if division_step > 0:
                #     if global_step % division_step == 0:
                #                 # division_step = (train_number // (1 * batch_size))
                #                 val_score, val_score2 = u2plevaluateval(net1, trainloader1, device)
                #                             # val_score, val_score2 = evaluate(net1, testloader, device)
                #                 # scheduler.step(val_score)

                #                 logging.info('training Dice score: {}'.format(val_score))
                #                 logging.info('training IOU score: {}'.format(val_score2))
                #                 print('learning rate:',optimizer.param_groups[0]['lr'],)
                            
                #                 print('epoch:',epoch)
                #                 # if global_step % 10 == 0:
                #                 writer.add_scalar('training Dice score',val_score, global_step=global_step)
                #                 writer.add_scalar('training IOU score',val_score2, global_step=global_step)
                if division_step > 0:
                    if global_step % division_step == 0:
        
                            val_score, val_score2 = u2plevaluateval(net1, val_loader, device)     
                            val_score = val_score
                            val_score2 = val_score2
                            # scheduler.step(val_score)
                            # print(len(val_loader))

                            logging.info('Validation Dice score: {}'.format(val_score))
                            logging.info('Validation IOU score: {}'.format(val_score2))
                            print('learning rate:',optimizer.param_groups[0]['lr'],)                  
                            print('epoch:',epoch)
                            # if global_step % 10 == 0:
                            writer.add_scalar('validation Dice score',val_score, global_step=global_step)
                            writer.add_scalar('validation IOU score',val_score2, global_step=global_step)
            #print('train confusion_matrix\n')
            #print(train_matrix)
                    # if global_step % (2*division_step) == 0:
                    if global_step % division_step == 0:
                            # val_score, val_score2 = evaluate(net1, val_loader, device)
                            val_score, val_score2 = u2plevaluate(net1, testloader, device)
                          
                            # dist.all_reduce(val_score)
                            # dist.all_reduce(val_score2)

                            val_score = val_score
                            val_score2 = val_score2
                            
                            # scheduler.step(val_score)
                            # print(len(testloader))

                            logging.info('Test Dice score: {}'.format(val_score))
                            logging.info('Test IOU score: {}'.format(val_score2))
                            print('learning rate:',optimizer.param_groups[0]['lr'],)
                        
                            print('epoch:',epoch)
                            
                            # if global_step % 10 == 0:
                            writer.add_scalar('test Dice score',val_score, global_step=global_step)
                            writer.add_scalar('test IOU score',val_score2, global_step=global_step)
                    if global_step % division_step == 0:
                            val_score, val_score2 = u2plevaluate(net1, testloader1, device)
                          
                            val_score = val_score
                            val_score2 = val_score2
                            
                            scheduler.step(val_score)
                            # print(len(testloader))

                            logging.info('Test Dice score: {}'.format(val_score))
                            logging.info('Test IOU score: {}'.format(val_score2))
                            print('learning rate:',optimizer.param_groups[0]['lr'],)
                        
                            print('epoch:',epoch)
                           
                          
                            # if global_step % 10 == 0:
                            writer.add_scalar('test247 Dice score',val_score, global_step=global_step)
                            writer.add_scalar('test247 IOU score',val_score2, global_step=global_step)
                   
        if save_checkpoint:
          if epoch > 500:
        #    if epoch%100 == 0:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)

            torch.save({
            'epoch': epoch,
            'model_state_dict': net1.module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch + 1)))

            # torch.save(backbone.module.state_dict(), str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch + 1)))
            logging.info(f'Checkpoint {epoch + 1} saved!')

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
# 设置随机数种子


if __name__ == '__main__':
    #python -m torch.distributed.launch --nproc_per_node 2 main.py
    args = parser.parse_args()
    setup_seed(3407)
    # pretext_model = torch.load('/home/wjc20/segmentation/byol/baseline/train3/fusion51epoch37.pth')
    # pretext_model = torch.load('/home/wjc20/segmentation/byol/newidea/unet/pth/prefusion/checkpoint_epoch57.pth')
    # pretext_model = torch.load('/home/wjc20/segmentation/byol/newidea/unet/pth/105/checkpoint_epoch10.pth')
    # pretext_model = torch.load('/home/wjc20/segmentation/byol/newidea/unet/pth/105/checkpoint_epoch10.pth')
    # pretext_model = torch.load('/home/wjc20/segmentation/pth/u2pl/9/augepoch25.pth')
    # pretext_model = torch.load('/home/wjc20/segmentation/byol/newidea/unet/pth/fusion/connew/checkpoint_epoch61.pth')
    # pretext_model = torch.load('/home/wjc20/segmentation/byol/newidea/unet/pth/con/con5/checkpoint_epoch66.pth')
    # pretext_model = torch.load('/home/wjc20/segmentation/byol/newidea/unet/pth/con/con3/checkpoint_epoch31.pth')
    # pretext_model = torch.load('/home/wjc20/segmentation/byol/newidea/unet/pth/con/con3/checkpoint_epoch31.pth')
    pretext_model = torch.load('/home/wjc20/segmentation/byol/newidea/unet/pth/fusion/connew/checkpoint_epoch64.pth')
    backbone = UNet(out_channels=2)
    backbone = backbone.cuda()
    backbone.load_state_dict(pretext_model['model_state_dict'],strict=False)
    # IPython.embed()
    teacher = UNet(out_channels=2)
    teacher = teacher.cuda()
    teacher.load_state_dict(pretext_model['model_state_dict'],strict=False)

    backbone = torch.nn.DataParallel(backbone)
    teacher = torch.nn.DataParallel(teacher)
    
    for p in teacher.parameters():
        p.requires_grad = False
    # for p in backbone.parameters():
    #     p.requires_grad = False

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logging.info(f'Using device {device}')

    
    try:
        train_net(net1=backbone,
                  net2=teacher,
                  epochs=EPOCHS,
                  batch_size=BATCH_SIZE,
                  learning_rate=LR,
                  device=device,
                  val_percent=0.7
                  )
    except KeyboardInterrupt:
        torch.save(backbone.module.state_dict(), '/home/wjc20/segmentation/byol/newidea/pth/epoch.pth') #2是864,1296预训练的batchsize4，  #1预训练的。1036-1555大小的 batch是3 70epoch
        logging.info('Saved interrupt')







# import os

# os.environ["CUDA_VISIBLE_DEVICES"] = '4,5'
# os.environ["OMP_NUM_THREADS"] = '32'
# import argparse
# import copy
# import logging
# import math
# import multiprocessing
# import random
# import sys
# import time  # /home/wjc20/segmentation/byol/newidea/unetseg.py
# from collections import OrderedDict
# from configparser import Interpolation
# from functools import wraps
# # import aug as trans
# from pathlib import Path

# import cv2
# import IPython
# import lightly.data as data
# import lightly.loss as loss
# import lightly.models as models
# import numpy as np
# import pytorch_lightning as pl
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torchvision
# from cv2 import INTER_NEAREST
# from PIL import Image
# from sklearn.metrics import confusion_matrix
# from sklearn.utils import shuffle
# from sympy import Ne, interpolate
# from torch import nn, optim
# from torch.utils.data import DataLoader, Dataset, random_split
# from torch.utils.tensorboard import SummaryWriter
# from torchvision import models
# from torchvision import transforms
# from torchvision import transforms as T
# from tqdm import tqdm

# import wandb
# from byol_pytorch import BYOL
# from dice_score import dice_loss
# from evaluate import evaluate, trainevaluate,u2plevaluate,u2plevaluateval
# from modules import *
# from unet import UNet
# from bdfset import UnlabelData,MyData,TestData,TestData1
# # from teacher import Net,CAM
# print(os.environ["CUDA_VISIBLE_DEVICES"])
# import torch.distributed as dist
# from torch.nn import SyncBatchNorm
# from torch.nn.parallel import DistributedDataParallel
# from torch.utils.data.distributed import DistributedSampler
# sys.path.append("..") 
# # from baseline.U2PL.u2pl.utils.dist_helper import setup_distributed
# # default `log_dir` is "runs" - we'll be more specific here
# writer = SummaryWriter('/home/wjc20/segmentation/byol/newidea/unet/log/retrain/ablation/embed') #

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# #dir_checkpoint = Path('./unetpth/onlyuse2modalcrossloss')
# dir_checkpoint = Path('/home/wjc20/segmentation/byol/newidea/unet/pth/ablation/embed')

# # a = pred_u_teacher
# # constants
# mean = [0.485, 0.456, 0.406]
# std = [0.229, 0.224, 0.225]
# BATCH_SIZE = 10
# moving_average_decay = 0.99
# EPOCHS     = 200
# LR         = 1e-3
# NUM_GPUS   = 2
# IMAGE_SIZE = 256
# IMAGE_EXTS = ['.jpg', '.png', '.jpeg']
# #NUM_WORKERS = multiprocessing.cpu_count()
# # pytorch lightning module
# import logging
# from os import listdir
# from os.path import splitext
# from pathlib import Path
# import yaml
# from sklearn.utils.multiclass import type_of_target
# parser = argparse.ArgumentParser(description="Semi-Supervised Semantic Segmentation")
# parser.add_argument("--config", type=str, default="config.yaml")
# parser.add_argument("--local_rank", type=int, default=0)
# parser.add_argument("--seed", type=int, default=0)
# parser.add_argument("--port", default=None, type=int)

# def train_net(net1,
#               net2,
#               device,
#               epochs=EPOCHS,
#               batch_size = BATCH_SIZE,
#               learning_rate = LR,
#               val_percent: float = 0.4,
#               save_checkpoint: bool = True,
#               amp: bool = False):
#     # cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)
#     dataset = MyData()
#     undataset = UnlabelData()
#     #Split into train / validation partitions
#     train_number = int(len(dataset) * val_percent)
#     tnumber = int(0.6 * (len(dataset)- train_number))
#     val_number = int(len(dataset) - train_number - tnumber)
#     number1 = int(len(undataset) * val_percent)
#     number2 = int((len(undataset) - number1))
#     # vnumber = len(dataset) - train_number - val_number
#     #train_number = int(len(dataset)) * low_threshold
#     #val_number = train_number
#     # train_set , val_set, vset = random_split(dataset, [train_number, val_number,vnumber], generator=torch.Generator().manual_seed(3407))
#     train_set , val_set ,tset = random_split(dataset, [train_number, val_number,tnumber], generator=torch.Generator().manual_seed(3407))
#     undataset, undataset1 = random_split(undataset, [number2, number1], generator=torch.Generator().manual_seed(3407))
#     # val_set, vset = random_split(val_set, [val_number*low_threshold, val_number-val_number*low_threshold], generator=torch.Generator().manual_seed(3407))
#     # Create data loaders
   
#     train_loader = DataLoader(train_set,batch_size=BATCH_SIZE,drop_last=True,shuffle = True)
#     val_loader = DataLoader(val_set,  batch_size=1,drop_last=True,shuffle = True)
#     unloader = DataLoader(undataset, batch_size=BATCH_SIZE,drop_last=True,shuffle = True)
#     #初始化logging

#     optimizer = optim.RMSprop(net1.parameters(), lr=LR, weight_decay=1e-5, momentum=0.8)
#     # optimizer = optim.RMSprop(net1.parameters(), lr=LR,momentum=high_threshold)
#     # optimizer = optim.RMSprop(net2.parameters(), lr=LR, weight_decay=1e-5, momentum=high_threshold)
#     scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor = 0.2, patience=3)  # goal: maximize Dice score  factor = 0.5,!!!!!!
#     grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    
#     global_step = 0
#     testset = TestData()
#     testset1 = TestData1()

#     train_number1 = int(len(train_set) * 0.1)
#     val_number1 = int((len(train_set) - train_number1))
#     train_set1 , val_set1 = random_split(train_set, [train_number1, val_number1], generator=torch.Generator().manual_seed(3407))
#     trainloader1 = DataLoader(train_set1,shuffle=True,batch_size=1,drop_last=True)
#     testloader = DataLoader(testset,shuffle=True,batch_size=1,drop_last=True)
#     testloader1 = DataLoader(testset1,shuffle=True,batch_size=1,drop_last=True)
#     criterion = nn.CrossEntropyLoss(weight = torch.from_numpy(np.array([0.2,0.8])).float(),ignore_index=2).cuda(device)
#     criterion1 = nn.CrossEntropyLoss(weight = torch.from_numpy(np.array([0.2,0.8])).float(),ignore_index=2).cuda(device)
    
#     for epoch in range(epochs):
#         net1.train()
#         count = 0

#         # high_threshold = 0.85
#         # low_threshold  = 0.15
#         # if epoch>15:
#         #     high_threshold = 0.8
#         #     low_threshold  = 0.2
#         # elif epoch>30:
#         #     high_threshold = 0.7
#         #     low_threshold  = 0.3
#         # elif epoch>50:
#         #     high_threshold = 0.8
#         #     low_threshold  = 0.2

#         high_threshold = 0.85
#         low_threshold  = 0.15
#         if epoch>15:
#             high_threshold = 0.75
#             low_threshold  = 0.25
#         elif epoch>30:
#             high_threshold = 0.65
#             low_threshold  = 0.35
#         epoch_loss = 0
#         train_matrix = 0
#         # train_loader.sampler.set_epoch(epoch)
#         # unloader.sampler.set_epoch(epoch)
#         loader_l_iter = iter(train_loader)
#         loader_u_iter = iter(unloader)
#         with tqdm(total=(2*len(train_loader)*BATCH_SIZE), desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
#         # for batch in tqdm(range(len(train_loader))):
#             for batch in range(len(train_loader)):
#                 # torch.cuda.empty_cache()
#                 try:
#                     images,labels = loader_l_iter.next()
#                     # IPython.embed()
#                 except:
#                     loader_l_iter = iter(train_loader)
#                     # IPython.embed()
#                     images,labels = loader_l_iter.next()
                
#                 labels = labels//200
#                 try: 
#                     image_u,imgu = loader_u_iter.next()
#                 except:
#                     loader_u_iter = iter(unloader)
#                     image_u,imgu = loader_u_iter.next()
#                 # b_size,chanel, h, w = labels.size()
                
#                 images = images.to(device=device, dtype=torch.float32)
#                 # images = images
#                 labels  = labels.to(device=device, dtype=torch.long)
#                 image_u = image_u.to(device=device, dtype=torch.float32)
#                 images = torch.squeeze(images)
#                 image_u = torch.squeeze(image_u)
#                 labels = torch.squeeze(labels,dim=1)
#                 # IPython.embed()
                
                #pred = F.one_hot(labels_pred.argmax(dim=1), 21).permute(0, 3, 1, 2).float()
                
                # zero = torch.zeros_like(labels)
                # one = torch.ones_like(labels)   
                # two = 2*torch.ones_like(labels)
                # zero = zero.to(dtype=torch.float32)     
                # one  = one.to(dtype=torch.float32)
                # two = two.to(dtype=torch.float32)
                
                # with torch.cuda.amp.autocast(enabled=amp):
                #     net2.eval()     #197 293 294与ema注释了也就是：
                    # with torch.no_grad():
                    #     out_t = net2(image_all)
                    # for s_params in zip(net1.parameters()):
                    
                        # print(s_params)
                    # for name, parms in net1.named_parameters():	
                    #     print(parms.requires_grad)
                
                    # pred_u_teacher = net2(image_u)
                    # abc = pred_u_teacher
                    # print(torch.min(pred_u_teacher),torch.max(pred_u_teacher),'step is {}'.format(global_step))
                    
                    # pred_u_teacher = F.interpolate(pred_u_teacher, (h, w), mode="bilinear", align_corners=True)
                    # pred_u_teacher = F.softmax(pred_u_teacher, dim=1)
                    # with open('/home/wjc20/segmentation/byol/newidea/unet/u2.txt', "a") as f:
                    #     s = torch.min(pred_u_teacher).item(), + torch.max(pred_u_teacher).item(), + torch.min(abc).item(),
                    #     + torch.max(abc).item() 
                    #     s = str(s) + '\n'
                        
                    #     torch.min(abc).item(),torch.max(abc).item(),'step is {}'.format(global_step)
                        # f.write(s)
                    #  'step is {}'.format(global_step))# torch.min(abc).item(),torch.max(abc).item(),
                    # if global_step==5:
                    #     a1 = net1.module.down3.maxpool_conv[1].double_conv[1].running_mean.clone()
                    #     b1 = net1.module.down3.maxpool_conv[1].double_conv[1].running_var.clone()
                    #     aa1 = net1.module.down3.maxpool_conv[1].double_conv[1].weight.clone()
                    #     a = net2.module.up2.conv.double_conv[1].running_mean.clone()
                    #     b = net2.module.up2.conv.double_conv[1].running_var.clone()
                    #     aa = net2.module.up2.conv.double_conv[1].weight.clone()
                    #     a2 = net2.module.up2.conv.double_conv[0].weight.clone()
                    # if global_step==25:
                    #     c1 = net1.module.down3.maxpool_conv[1].double_conv[1].running_mean.clone()
                    #     d1= net1.module.down3.maxpool_conv[1].double_conv[1].running_var.clone()
                    #     cc1 = net1.module.down3.maxpool_conv[1].double_conv[1].weight.clone()
                    #     c = net2.module.up2.conv.double_conv[1].running_mean.clone()
#                     #     d = net2.module.up2.conv.double_conv[1].running_var.clone()
#                     #     cc = net2.module.up2.conv.double_conv[1].weight.clone()
#                     #     c2 = net2.module.up2.conv.double_conv[0].weight.clone()
#                     label_u = torch.where(pred_u_teacher > high_threshold, one, pred_u_teacher)
#                     label_u_dice = label_u
#                     # IPython.embed()
#                     label_u_dice = torch.where(pred_u_teacher <= high_threshold, zero, label_u_dice)
#                     label_u = torch.where(label_u < low_threshold, zero, label_u)
#                     # IPython.embed()
#                     # label_u_dice = label_u
#                     # label_u_dice = torch.where(torch.gt(label_u_dice, low_threshold) & torch.lt(label_u_dice, high_threshold), zero, label_u_dice)

#                     label_u = torch.where(torch.gt(label_u, low_threshold) & torch.lt(label_u, high_threshold), two, label_u)
#                     label_u = label_u[:,1].to(dtype=torch.long)
#                     label_u_dice = label_u_dice[:,1].to(dtype=torch.long)
#                     # IPython.embed()
#                     if torch.min(label_u_dice)<0:
#                         IPython.embed()
#                         count =+1
#                         continue
                    
#                     num_labeled = len(images)
#                     image_all = torch.cat((images,image_u))
#                     # labels_pred = net1(images)
#                     outs = net1(image_all)
#                     pred_l, pred_u = outs[:num_labeled], outs[num_labeled:]  
#                     # with open('/home/wjc20/segmentation/byol/newidea/unet/u2pl.txt', "a") as f:
#                     #     s = torch.min(pred_l).item(), +  torch.max(pred_l).item(),+torch.min(pred_u).item(),torch.max(pred_u).item()
#                     #     s = str(s) + '\n'
#                     # #     torch.min(abc).item(),torch.max(abc).item(),'step is {}'.format(global_step)
#                     #     f.write(s)
#                     # print(torch.min(pred_l).item(),torch.max(pred_l).item(),'step is {}'.format(global_step))
#                     # print(torch.min(pred_u).item(),torch.max(pred_u).item(),'un step is {}'.format(global_step))
#                     # print(torch.sum(label_u_dice==1))
#                     # outs = net1(images)
#                     # pred_l = outs
#                     # pred_u = outs
#                     # for name, parms in net1.named_parameters():	
#                     #     print(parms.requires_grad)
#                     # for name, parms in net2.named_parametexiters():	
#                     #     print(parms.requires_grad)
                   
                               
#                     # loss1
                    
#                     labels = torch.squeeze(labels)
#                     sup_loss1 = criterion1(pred_l, labels) 
#                     soft_pred = F.softmax(pred_l,dim=1).float()
#                     labeldice = copy.deepcopy(labels)
#                     labeldice[labeldice==2]=0
#                     labeldice  = labeldice.to(device=device, dtype=torch.long)
#                     labeldice = F.one_hot(labeldice, 2).permute(0, 3 ,1, 2).float() 
                    
#                     sup_loss2 = dice_loss(soft_pred[:,...],
#                                      labeldice[:,...],
#                                        multiclass=False)
#                     sup_loss = sup_loss1 +  sup_loss2
#                     #sup_loss.requires_grad
# # torch.sum(label_u==1),torch.sum(label_u==2),torch.sum(label_u),torch.min(label_u),torch.max(label_u)  torch.sum(labels==1),torch.sum(labels)
# # torch.sum(label_dice==1),torch.sum(label_dice==2),torch.sum(label_dice)  torch.sum(low_threshold<label_u&&label_u>high_threshold)
#                                         #loss2
#                     net2.train()
#                     with torch.no_grad():
#                         out_t = net2(image_all)
#                     unsup_loss1 = criterion(pred_u, label_u)
#                     unsoft_pred = F.softmax(pred_u,dim=1).float()
                    
#                     # if global_step==4:
#                     #     IPython.embed()
#                     # if global_step>10:
#                     #     IPython.embed()
#                     # print('\n',torch.max(label_u_dice),torch.min(label_u_dice))
                    
#                     label_dice = F.one_hot(label_u_dice, 2).permute(0, 3 ,1, 2).float() 

#                     unsup_loss2 = dice_loss(unsoft_pred[:,:,...],
#                                      label_dice[:,:, ...],
#                                        multiclass=False)
#                     unsup_loss = unsup_loss1 + unsup_loss2
#                     # print(sup_loss1.item(),sup_loss2.item(),unsup_loss1.item(),unsup_loss2.item())
#                     loss = sup_loss +  unsup_loss
#                     # IPython.embed()
#                     # if global_step ==4000:
#                     #     IPython.embed()
#                     # if global_step ==10000:
#                     #     IPython.embed()
#                     # if global_step ==15000:
#                     #     IPython.embed()

#                     # print(sup_loss1,sup_loss2,unsup_loss1,unsup_loss2)
                    
#                     # net2.train()
#                     # unsuploss:
#                     # with torch.no_grad():
#                     #     out_t = net2(image_all)
#                     #     pred_all_teacher = out_t
#                     #     prob_all_teacher = F.softmax(pred_all_teacher, dim=1)
#                     #     prob_l_teacher, prob_u_teacher = (
#                     #         prob_all_teacher[:num_labeled],
#                     #         prob_all_teacher[num_labeled:],
#                     #     )

#                     #     pred_u_teacher = pred_all_teacher[num_labeled:]
                    

                    
#                     if global_step % 20 == 0:
#                         writer.add_scalar('train crossloss',sup_loss1.item(),global_step=global_step)
#                         writer.add_scalar('train diceloss',sup_loss2.item(),global_step=global_step)
#                         writer.add_scalar('train unlabel crossloss',unsup_loss1.item(),global_step=global_step)
#                         writer.add_scalar('train unlabel diceloss',unsup_loss2.item(),global_step=global_step)
                      
#                 # IPython.embed()
#                 # optimizer.zero_grad(set_to_none=True)
#                 optimizer.zero_grad()
#                 grad_scaler.scale(loss).backward()
#                 grad_scaler.step(optimizer)
#                 grad_scaler.update()

#                 pbar.update(2*images.shape[0])
#                 global_step += 1
#                 with torch.no_grad():
#                     iterl = global_step 
#                     ema_decay = moving_average_decay #min(1-1/((iterl-len(train_loader))+1.1),moving_average_decay)
#                     for t_params, s_params in zip(net2.parameters(),net1.parameters()):
#                         t_params.data = ema_decay*t_params.data + (1 - ema_decay) * s_params.data
#                 # IPython.embed()
#                 pbar.set_postfix(**{'loss (batch)': loss.item()})
        
#                 division_step = len(train_loader)#(train_number // (1 * batch_size))
#                 if division_step > 0:
#                     if global_step % division_step == 0:
#                                 # division_step = (train_number // (1 * batch_size))
#                                 val_score, val_score2 = u2plevaluateval(net1, trainloader1, device)
#                                             # val_score, val_score2 = evaluate(net1, testloader, device)
#                                 # scheduler.step(val_score)

#                                 logging.info('training Dice score: {}'.format(val_score))
#                                 logging.info('training IOU score: {}'.format(val_score2))
#                                 print('learning rate:',optimizer.param_groups[0]['lr'],)
                            
#                                 print('epoch:',epoch)
#                                 # if global_step % 10 == 0:
#                                 writer.add_scalar('training Dice score',val_score, global_step=global_step)
#                                 writer.add_scalar('training IOU score',val_score2, global_step=global_step)
#                 if division_step > 0:
#                     if global_step % division_step == 0:
#                             # histograms = {}
#                             # for tag, value in net1.named_parameters():
#                             #     tag = tag.replace('/', '.')
#                             #     histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
#                             #     histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())
#                             # print('the epoch ignore step is:{}'.format(count))
#                             val_score, val_score2 = u2plevaluateval(net1, val_loader, device)
#                             # print('the epoch ignore step is:{}'.format(count))
                           
#                             # dist.all_reduce(val_score)
#                             # dist.all_reduce(val_score2)
                       
#                             val_score = val_score
#                             val_score2 = val_score2
#                             # scheduler.step(val_score)
#                             # print(len(val_loader))

#                             logging.info('Validation Dice score: {}'.format(val_score))
#                             logging.info('Validation IOU score: {}'.format(val_score2))
#                             print('learning rate:',optimizer.param_groups[0]['lr'],)                  
#                             print('epoch:',epoch)
#                             # if global_step % 10 == 0:
#                             writer.add_scalar('validation Dice score',val_score, global_step=global_step)
#                             writer.add_scalar('validation IOU score',val_score2, global_step=global_step)
#             #print('train confusion_matrix\n')
#             #print(train_matrix)
#                     # if global_step % (2*division_step) == 0:
#                     if global_step % division_step == 0:
#                             # histograms = {}
#                             # for tag, value in net1.named_parameters():
#                             #     tag = tag.replace('/', '.')
#                             #     histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
#                             #     histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())
                      

#                             # val_score, val_score2 = evaluate(net1, val_loader, device)
#                             val_score, val_score2 = u2plevaluate(net1, testloader, device)
                          
#                             # dist.all_reduce(val_score)
#                             # dist.all_reduce(val_score2)

#                             val_score = val_score
#                             val_score2 = val_score2
                            
#                             # scheduler.step(val_score)
#                             # print(len(testloader))

#                             logging.info('Test Dice score: {}'.format(val_score))
#                             logging.info('Test IOU score: {}'.format(val_score2))
#                             print('learning rate:',optimizer.param_groups[0]['lr'],)
                        
#                             print('epoch:',epoch)
                            
#                             # if global_step % 10 == 0:
#                             writer.add_scalar('test Dice score',val_score, global_step=global_step)
#                             writer.add_scalar('test IOU score',val_score2, global_step=global_step)
#                     if global_step % division_step == 0:
#                             # histograms = {}
#                             # for tag, value in net1.named_parameters():
#                             #     tag = tag.replace('/', '.')
#                             #     histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
#                             #     histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())
                      

#                             # val_score, val_score2 = evaluate(net1, val_loader, device)
#                             val_score, val_score2 = u2plevaluate(net1, testloader1, device)
                          
#                             # dist.all_reduce(val_score)
#                             # dist.all_reduce(val_score2)

#                             val_score = val_score
#                             val_score2 = val_score2
                            
#                             scheduler.step(val_score)
#                             # print(len(testloader))

#                             logging.info('Test Dice score: {}'.format(val_score))
#                             logging.info('Test IOU score: {}'.format(val_score2))
#                             print('learning rate:',optimizer.param_groups[0]['lr'],)
                        
#                             print('epoch:',epoch)
                           
                          
#                             # if global_step % 10 == 0:
#                             writer.add_scalar('test247 Dice score',val_score, global_step=global_step)
#                             writer.add_scalar('test247 IOU score',val_score2, global_step=global_step)
                   
#         if save_checkpoint:
#           if epoch > 500:
#         #    if epoch%100 == 0:
#             Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)

#             torch.save({
#             'epoch': epoch,
#             'model_state_dict': net1.module.state_dict(),
#             'optimizer_state_dict': optimizer.state_dict(),
#             'loss': loss,
#             }, str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch + 1)))

#             # torch.save(backbone.module.state_dict(), str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch + 1)))
#             logging.info(f'Checkpoint {epoch + 1} saved!')

# def setup_seed(seed):
#      torch.manual_seed(seed)
#      torch.cuda.manual_seed_all(seed)
#      np.random.seed(seed)
#      random.seed(seed)
#      torch.backends.cudnn.deterministic = True
# # 设置随机数种子


# if __name__ == '__main__':
#     #python -m torch.distributed.launch --nproc_per_node 2 main.py
#     args = parser.parse_args()
#     setup_seed(3407)
#     # pretext_model = torch.load('/home/wjc20/segmentation/byol/baseline/train3/fusion51epoch37.pth')
#     # pretext_model = torch.load('/home/wjc20/segmentation/byol/newidea/unet/pth/prefusion/checkpoint_epoch57.pth')
#     # pretext_model = torch.load('/home/wjc20/segmentation/byol/newidea/unet/pth/105/checkpoint_epoch10.pth')
#     # pretext_model = torch.load('/home/wjc20/segmentation/byol/newidea/unet/pth/105/checkpoint_epoch10.pth')
#     # pretext_model = torch.load('/home/wjc20/segmentation/pth/u2pl/9/augepoch25.pth')
#     # pretext_model = torch.load('/home/wjc20/segmentation/byol/newidea/unet/pth/fusion/connew/checkpoint_epoch61.pth')
#     # pretext_model = torch.load('/home/wjc20/segmentation/byol/newidea/unet/pth/con/con5/checkpoint_epoch66.pth')
#     # pretext_model = torch.load('/home/wjc20/segmentation/byol/newidea/unet/pth/con/con3/checkpoint_epoch31.pth')
#     # pretext_model = torch.load('/home/wjc20/segmentation/byol/newidea/unet/pth/con/con3/checkpoint_epoch31.pth')
#     pretext_model = torch.load('/home/wjc20/segmentation/byol/newidea/unet/pth/fusion/connew/checkpoint_epoch64.pth')
#     backbone = UNet(out_channels=2)
#     backbone = backbone.cuda()
#     backbone.load_state_dict(pretext_model['model_state_dict'],strict=False)
    
#     teacher = UNet(out_channels=2)
#     teacher = teacher.cuda()
#     teacher.load_state_dict(pretext_model['model_state_dict'],strict=False)

#     backbone = torch.nn.DataParallel(backbone)
#     teacher = torch.nn.DataParallel(teacher)
#     IPython.embed()
#     for p in teacher.parameters():
#         p.requires_grad = False
#     # for p in backbone.parameters():
#     #     p.requires_grad = False

#     logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
#     logging.info(f'Using device {device}')

    
#     try:
#         train_net(net1=backbone,
#                   net2=teacher,
#                   epochs=EPOCHS,
#                   batch_size=BATCH_SIZE,
#                   learning_rate=LR,
#                   device=device,
#                   val_percent=0.7
#                   )
#     except KeyboardInterrupt:
#         torch.save(backbone.module.state_dict(), '/home/wjc20/segmentation/byol/newidea/pth/epoch.pth') #2是864,1296预训练的batchsize4，  #1预训练的。1036-1555大小的 batch是3 70epoch
#         logging.info('Saved interrupt')










# import os

# os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
# os.environ["OMP_NUM_THREADS"] = '32'
# import argparse
# import copy
# import logging
# import math
# import multiprocessing
# import random
# import sys
# import time  # /home/wjc20/segmentation/byol/newidea/unetseg.py
# from collections import OrderedDict
# from configparser import Interpolation
# from functools import wraps
# # import aug as trans
# from pathlib import Path

# import cv2
# import IPython
# import lightly.data as data
# import lightly.loss as loss
# import lightly.models as models
# import numpy as np
# import pytorch_lightning as pl
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torchvision
# from cv2 import INTER_NEAREST
# from PIL import Image
# from sklearn.metrics import confusion_matrix
# from sklearn.utils import shuffle
# from sympy import Ne, interpolate
# from torch import nn, optim
# from torch.utils.data import DataLoader, Dataset, random_split
# from torch.utils.tensorboard import SummaryWriter
# from torchvision import models
# from torchvision import transforms
# from torchvision import transforms as T
# from tqdm import tqdm




# import wandb
# from byol_pytorch import BYOL
# from dice_score import dice_loss
# from evaluate import evaluate, trainevaluate
# from modules import *
# from unet import UNet

# # from teacher import Net,CAM
# print(os.environ["CUDA_VISIBLE_DEVICES"])
# import torch.distributed as dist
# from torch.nn import SyncBatchNorm
# from torch.nn.parallel import DistributedDataParallel
# from torch.utils.data.distributed import DistributedSampler
# sys.path.append("..") 
# from baseline.U2PL.u2pl.utils.dist_helper import setup_distributed
# # default `log_dir` is "runs" - we'll be more specific here
# writer = SummaryWriter('/home/wjc20/segmentation/byol/newidea/unet/log/u2pllog') #

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# #dir_checkpoint = Path('./unetpth/onlyuse2modalcrossloss')
# dir_checkpoint = Path('/home/wjc20/segmentation/byol/newidea/unet/pth/u2pl/1')

# # a = pred_u_teacher
# # constants
# mean = [0.485, 0.456, 0.406]
# std = [0.229, 0.224, 0.225]
# BATCH_SIZE = 4
# moving_average_decay = high_threshold9
# EPOCHS     = 200
# LR         = 1e-3
# NUM_GPUS   = 2
# IMAGE_SIZE = 256
# IMAGE_EXTS = ['.jpg', '.png', '.jpeg']
# #NUM_WORKERS = multiprocessing.cpu_count()
# # pytorch lightning module
# import logging
# from os import listdir
# from os.path import splitext
# from pathlib import Path

# from sklearn.utils.multiclass import type_of_target
# parser = argparse.ArgumentParser(description="Semi-Supervised Semantic Segmentation")
# parser.add_argument("--config", type=str, default="config.yaml")
# parser.add_argument("--local_rank", type=int, default=0)
# parser.add_argument("--seed", type=int, default=0)
# parser.add_argument("--port", default=None, type=int)

# class EMA():
#     def __init__(self, beta):
#         super().__init__()
#         self.beta = beta

#     def update_average(self, old, new):
#         if old is None:
#             return new
#         return old * self.beta + (1 - self.beta) * new
# def update_moving_average(ema_updater, ma_model, current_model):
#     for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
#         old_weight, up_weight = ma_params.data, current_params.data
#         ma_params.data = ema_updater.update_average(old_weight, up_weight)
# class UnlabelData(Dataset):
#     def __init__(self):
#         self.sample_list1 = list()
       
#         f = open('/home/wjc20/segmentation/byol/newidea/txt/unsup.txt', 'r')
#         # f = open('/home/wjc20/segmentation/byol/newidea/txt/105fusionimg.txt', 'r')
#         line1 = f.readlines()
#         for line in line1:
#             self.sample_list1.append(line.strip())
#         f.close()
  
        
#         # self.transform = trans.Compose([
#         #     trans.RandomGaussianBlur(),
#         #    # transforms.ToTensor(),
#         #     transforms.Normalize(mean=mean, std=std)])
#     def __len__(self):
#         return (len(self.sample_list1))

#     def __getitem__(self, index):
#         item1= self.sample_list1[index]
 
#         img1 = cv2.imread(item1)
        
#         img1= cv2.resize(img1, (384,576), interpolation=INTER_NEAREST)
#         img1 = np.transpose(img1, (2, 0, 1))
        
#         return (img1)#1是彩色图像，2是标签图。
# class MyData(Dataset):
#     def __init__(self):
#         self.sample_list1 = list()
#         self.sample_list2 = list()
#         self.sample_list3 = list()
#         # f = open('/home/wjc20/segmentation/byol/newidea/txt/51fusionimg.txt', 'r')
#         f = open('/home/wjc20/segmentation/byol/newidea/txt/105fusionimg.txt', 'r')
#         line1 = f.readlines()
#         for line in line1:
#             self.sample_list1.append(line.strip())
#         f.close()
#         # f = open('/home/wjc20/segmentation/byol/newidea/txt/51fusionlabel.txt', 'r')
#         f = open('/home/wjc20/segmentation/byol/newidea/txt/105fusionlabel.txt', 'r')
#         line2 = f.readlines()
#         for linen in line2:
#             self.sample_list2.append(linen.strip())
#         f.close()
#         # f = open('/home/wjc20/segmentation/byol/newidea/618label.txt', 'r')
#         # line3 = f.readlines()
#         # for line in line3:
#         #     self.sample_list3.append(line.strip())
#         # f.close()
        
        
#         # self.transform = trans.Compose([
#         #     trans.RandomGaussianBlur(),
#         #    # transforms.ToTensor(),
#         #     transforms.Normalize(mean=mean, std=std)])
#     def __len__(self):
#         return (len(self.sample_list1))

#     def __getitem__(self, index):
#         item1= self.sample_list1[index]
#         item2= self.sample_list2[index]
#         # item3= self.sample_list3[index]

#         img1 = cv2.imread(item1)
        
#         img1= cv2.resize(img1, (384,576), interpolation=INTER_NEAREST)
#         img1 = np.transpose(img1, (2, 0, 1))


#         img2 = cv2.imread(item2, cv2.IMREAD_GRAYSCALE)
#         img2 = cv2.resize(img2, (384,576), interpolation=INTER_NEAREST)
#         img2 = np.expand_dims(img2, axis=2)
#         img2 = np.transpose(img2, (2, 0, 1))

#         # img3 = cv2.imread(item3,cv2.IMREAD_GRAYSCALE )
#         # img3 = cv2.resize(img3, (384,576), interpolation=INTER_NEAREST)
#         # img3 = np.expand_dims(img3, axis=2)
#         # img3 = np.transpose(img3, (2, 0, 1))
        
#         return (img1,img2)#1是彩色图像，2是标签图。

# class TestData(Dataset):
#     def __init__(self):
#         self.sample_list1 = list()
#         self.sample_list2 = list()

#         f = open('/home/wjc20/segmentation/byol/newidea/txt/247img.txt', 'r')  #测试集
#         line1 = f.readlines()
#         for line in line1:
#             self.sample_list1.append(line.strip())
#         f.close()

#         f = open('/home/wjc20/segmentation/byol/newidea/txt/247label.txt', 'r')
#         line1 = f.readlines()
#         for line in line1:
#             self.sample_list2.append(line.strip())
#         f.close()

#     def __len__(self):
#         return (len(self.sample_list1))

#     def __getitem__(self, index):
#         item1= self.sample_list1[index]
        
#         item2= self.sample_list2[index]
        
#         img1 = cv2.imread(item1)
#         img2 = cv2.imread(item2, cv2.IMREAD_GRAYSCALE)

#         # h, w, c= img1.shape
#         # if h<w:
#         #     img1 = np.rot90(img1, 1)
#         #     img2 = np.rot90(img2, 1)
#         high, wid, c = img1.shape
#         img1= cv2.resize(img1, (384,576), interpolation=INTER_NEAREST)
#         img1 = np.transpose(img1, (2, 0, 1))

#         img2 = cv2.resize(img2, (384,576), interpolation=INTER_NEAREST)
#         img2 = np.expand_dims(img2, axis=2)
   
        
#         img2 = np.transpose(img2, (2, 0, 1))

        
#         return (img1,img2)#1是彩色图像，2是标签图。
# class TestData1(Dataset):
#     def __init__(self):
#         self.sample_list1 = list()
#         self.sample_list2 = list()

#         f = open('/home/wjc20/segmentation/byol/baseline/txt/alltestimg.txt', 'r')  #测试集
#         line1 = f.readlines()
#         for line in line1:
#             self.sample_list1.append(line.strip())
#         f.close()

#         f = open('/home/wjc20/segmentation/byol/baseline/txt/alltestlabel.txt', 'r')
#         line1 = f.readlines()
#         for line in line1:
#             self.sample_list2.append(line.strip())
#         f.close()

#     def __len__(self):
#         return (len(self.sample_list1))

#     def __getitem__(self, index):
#         item1= self.sample_list1[index]
        
#         item2= self.sample_list2[index]
        
#         img1 = cv2.imread(item1)
#         img2 = cv2.imread(item2, cv2.IMREAD_GRAYSCALE)

#         # h, w, c= img1.shape
#         # if h<w:
#         #     img1 = np.rot90(img1, 1)
#         #     img2 = np.rot90(img2, 1)
#         high, wid, c = img1.shape
#         img1= cv2.resize(img1, (384,576), interpolation=INTER_NEAREST)
#         img1 = np.transpose(img1, (2, 0, 1))

#         img2 = cv2.resize(img2, (384,576), interpolation=INTER_NEAREST)
#         img2 = np.expand_dims(img2, axis=2)
      
        
#         img2 = np.transpose(img2, (2, 0, 1))

        
#         return (img1,img2)#1是彩色图像，2是标签图。

# def train_net(net1,
#               net2,
#               device,
#               epochs=EPOCHS,
#               batch_size = BATCH_SIZE,
#               learning_rate = LR,
#               val_percent: float = 0.4,
#               save_checkpoint: bool = True,
#               amp: bool = False):
#     dataset = MyData()
#     undataset = UnlabelData()
#     #Split into train / validation partitions
#     train_number = int(len(dataset) * val_percent)
#     tnumber = int(high_threshold * (len(dataset)- train_number))
#     val_number = int(len(dataset) - train_number - tnumber)
#     number1 = int(len(undataset) * val_percent)
#     number2 = int((len(undataset) - number1))
#     # vnumber = len(dataset) - train_number - val_number
#     #train_number = int(len(dataset)) * low_threshold
#     #val_number = train_number
#     # train_set , val_set, vset = random_split(dataset, [train_number, val_number,vnumber], generator=torch.Generator().manual_seed(3407))
#     train_set , val_set ,tset = random_split(dataset, [train_number, val_number,tnumber], generator=torch.Generator().manual_seed(3407))
#     undataset, undataset1 = random_split(undataset, [number2, number1], generator=torch.Generator().manual_seed(3407))
#     # val_set, vset = random_split(val_set, [val_number*low_threshold, val_number-val_number*low_threshold], generator=torch.Generator().manual_seed(3407))
#     # Create data loaders
#     sampler1 = DistributedSampler(train_set)
#     sampler2 = DistributedSampler(val_set)
#     sampler3 = DistributedSampler(undataset)
#     train_loader = DataLoader(train_set, shuffle=False, batch_size=BATCH_SIZE,drop_last=True,sampler=sampler1)
#     val_loader = DataLoader(val_set, shuffle=False, batch_size=1,drop_last=True,sampler=sampler2)
#     unloader = DataLoader(undataset, shuffle=False, batch_size=BATCH_SIZE,drop_last=True,sampler=sampler3)
#     #初始化logging

#     optimizer = optim.RMSprop(net1.parameters(), lr=LR, weight_decay=1e-5, momentum=high_threshold)
#     # optimizer = optim.RMSprop(net2.parameters(), lr=LR, weight_decay=1e-5, momentum=high_threshold)
#     scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor = 0.5, patience=2)  # goal: maximize Dice score  factor = 0.5,!!!!!!
#     grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
#     #criterion = nn.CrossEntropyLoss(ignore_index=255)
#     global_step = 0
#     testset = TestData()
#     testset1 = TestData1()
    
#     testloader = DataLoader(testset,shuffle=True,batch_size=1,drop_last=True)
#     testloader1 = DataLoader(testset1,shuffle=True,batch_size=1,drop_last=True)
    
#     for epoch in range(epochs):
#         net1.train()
#         epoch_loss = 0
#         train_matrix = 0
#         train_loader.sampler.set_epoch(epoch)
#         unloader.sampler.set_epoch(epoch)
#         loader_l_iter = iter(train_loader)
#         loader_u_iter = iter(unloader)
#         # for p in net2.parameters():
#         #      p.requires_grad = False
#         with tqdm(total=(2*len(train_loader)*NUM_GPUS*BATCH_SIZE), desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
#         # for batch in tqdm(range(len(train_loader))):
#             for batch in range(len(train_loader)):
#                 # images = batch[0]
#                 # labels = batch[1]//200
#                 images,labels = loader_l_iter.next()
#                 labels = labels//200
#                 image_u = loader_u_iter.next()
#                 b_size,chanel, h, w = labels.size()
#                 images = images.to(device=device, dtype=torch.float32)
#                 labels  = labels.to(device=device, dtype=torch.long)
#                 image_u = image_u.to(device=device, dtype=torch.float32)
                
#                 criterion = nn.CrossEntropyLoss(weight = torch.from_numpy(np.array([0.4,0.6])).float(),ignore_index=2).cuda(device)
#                 criterion1 = nn.CrossEntropyLoss(weight = torch.from_numpy(np.array([0.4,0.6])).float()).cuda(device)
#                 #pred = F.one_hot(labels_pred.argmax(dim=1), 21).permute(0, 3, 1, 2).float()
                
#                 zero = torch.zeros_like(labels)
#                 one = torch.ones_like(labels)   
#                 two = 2*torch.ones_like(labels)
#                 zero = zero.to(dtype=torch.float32)     
#                 one  = one.to(dtype=torch.float32)
#                 two = two.to(dtype=torch.float32)
                
#                 with torch.cuda.amp.autocast(enabled=amp):
#                     net2.eval()
#                     pred_u_teacher = net2(image_u)
#                     pred_u_teacher = F.interpolate(pred_u_teacher, (h, w), mode="bilinear", align_corners=True)
#                     pred_u_teacher = F.softmax(pred_u_teacher, dim=1)

#                     label_u = torch.where(pred_u_teacher > high_threshold, one, pred_u_teacher)
#                     label_u_dice = label_u
#                     label_u_dice = torch.where(pred_u_teacher < high_threshold1, zero, label_u_dice)
#                     label_u = torch.where(label_u < low_threshold, zero, label_u)
#                     # label_u_dice = label_u
#                     # label_u_dice = torch.where(torch.gt(label_u_dice, low_threshold) & torch.lt(label_u_dice, high_threshold), zero, label_u_dice)

#                     label_u = torch.where(torch.gt(label_u, 0.001) & torch.lt(label_u, high_threshold1), two, label_u)
#                     label_u = label_u[:,1].to(dtype=torch.long)
#                     label_u_dice = label_u_dice[:,1].to(dtype=torch.long)
                    
#                     num_labeled = len(images)
#                     image_all = torch.cat((images,image_u))
#                     # labels_pred = net1(images)
#                     outs = net1(image_all)
#                     # for name, parms in net1.named_parameters():	
#                     #     print(parms.requires_grad)
#                     # for name, parms in net2.named_parameters():	
#                     #     print(parms.requires_grad)
                   
#                     pred_l, pred_u = outs[:num_labeled], outs[num_labeled:]               
#                     # loss1
                    
#                     labels = torch.squeeze(labels)
#                     sup_loss1 = criterion1(pred_l, labels) 
#                     soft_pred = F.softmax(pred_l,dim=1).float()
#                     labels = F.one_hot(labels, 2).permute(0, 3 ,1, 2).float() 
                    
#                     sup_loss2 = dice_loss(soft_pred[:,1,...],
#                                      labels[:, 1, ...],
#                                        multiclass=False)
#                     sup_loss = sup_loss1 + sup_loss2
#                     #sup_loss.requires_grad
# # torch.sum(label_u==1),torch.sum(label_u==2),torch.sum(label_u),torch.min(label_u),torch.max(label_u)  torch.sum(labels==1),torch.sum(labels)
# # torch.sum(label_dice==1),torch.sum(label_dice==2),torch.sum(label_dice)  torch.sum(low_threshold<label_u&&label_u>high_threshold)
#                                         #loss2
#                     unsup_loss1 = criterion(pred_u, label_u)
#                     unsoft_pred = F.softmax(pred_u,dim=1).float()
#                     # if global_step==4:
#                     #     IPython.embed()
#                     # if global_step>10:
#                     #     IPython.embed()
#                     # print('\n',torch.max(label_u_dice),torch.min(label_u_dice))
#                     if torch.min(label_u_dice)<0:
#                         IPython.embed()
#                     label_dice = F.one_hot(label_u_dice, 2).permute(0, 3 ,1, 2).float() 
#                     unsup_loss2 = dice_loss(unsoft_pred[:,1,...],
#                                      label_dice[:, 1, ...],
#                                        multiclass=False)
#                     unsup_loss = unsup_loss1 + unsup_loss2
#                     loss = sup_loss #+ unsup_loss
#                     # print(sup_loss1,sup_loss2,unsup_loss1,unsup_loss2)
                    
#                     # net2.train()
#                     # unsuploss:
#                     # with torch.no_grad():
#                     #     out_t = net2(image_all)
#                     #     pred_all_teacher = out_t
#                     #     prob_all_teacher = F.softmax(pred_all_teacher, dim=1)
#                     #     prob_l_teacher, prob_u_teacher = (
#                     #         prob_all_teacher[:num_labeled],
#                     #         prob_all_teacher[num_labeled:],
#                     #     )

#                     #     pred_u_teacher = pred_all_teacher[num_labeled:]

                    
#                     if global_step % 3 == 0:
#                         writer.add_scalar('train crossloss',sup_loss1.item(),global_step=global_step)
#                         writer.add_scalar('train diceloss',sup_loss2.item(),global_step=global_step)
#                         writer.add_scalar('train unlabel crossloss',unsup_loss1.item(),global_step=global_step)
#                         writer.add_scalar('train unlabel diceloss',unsup_loss2.item(),global_step=global_step)
                      

#                 # optimizer.zero_grad(set_to_none=True)
#                 optimizer.zero_grad()
                        
#                 grad_scaler.scale(loss).backward()
#                 # loss.backward()
#                 grad_scaler.step(optimizer)
#                 grad_scaler.update()

#                 pbar.update(2*NUM_GPUS*images.shape[0])
#                 global_step += 1
#                 # with torch.no_grad():
#                 #     iterl = global_step 
#                 #     ema_decay = high_threshold9#min(1-1/(iterl+1),moving_average_decay)
#                 #     for t_params, s_params in zip(net2.parameters(),net1.parameters()):
#                 #         t_params.data = ema_decay*t_params.data + (1 - ema_decay) * s_params.data
              
#                 pbar.set_postfix(**{'loss (batch)': loss.item()})
        
#                 division_step = len(train_loader)#(train_number // (1 * batch_size))
#                 if division_step > 0:
#                     if global_step % division_step == 0:
#                             histograms = {}
#                             for tag, value in net1.named_parameters():
#                                 tag = tag.replace('/', '.')
#                                 histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
#                                 histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())
                      
#                             val_score, val_score2 = evaluate(net1, val_loader, device)
                    
                           
#                             # dist.all_reduce(val_score)
#                             # dist.all_reduce(val_score2)
                       
#                             val_score = val_score
#                             val_score2 = val_score2
#                             scheduler.step(val_score)
#                             print(len(val_loader))

#                             logging.info('Validation Dice score: {}'.format(val_score))
#                             logging.info('Validation IOU score: {}'.format(val_score2))
#                             print('learning rate:',optimizer.param_groups[0]['lr'],)                  
#                             print('epoch:',epoch)
#                             # if global_step % 10 == 0:
#                             writer.add_scalar('validation Dice score',val_score, global_step=global_step)
#                             writer.add_scalar('validation IOU score',val_score2, global_step=global_step)
#             #print('train confusion_matrix\n')
#             #print(train_matrix)
#                     # if global_step % (2*division_step) == 0:
#                     if global_step % division_step == 0:
#                             histograms = {}
#                             for tag, value in net1.named_parameters():
#                                 tag = tag.replace('/', '.')
#                                 histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
#                                 histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())
                      

#                             # val_score, val_score2 = evaluate(net1, val_loader, device)
#                             val_score, val_score2 = evaluate(net1, testloader, device)
                          
#                             # dist.all_reduce(val_score)
#                             # dist.all_reduce(val_score2)

#                             val_score = val_score
#                             val_score2 = val_score2
                            
#                             scheduler.step(val_score)
#                             print(len(testloader))

#                             logging.info('Test Dice score: {}'.format(val_score))
#                             logging.info('Test IOU score: {}'.format(val_score2))
#                             print('learning rate:',optimizer.param_groups[0]['lr'],)
                        
#                             print('epoch:',epoch)
#                             IPython.embed()
#                             # if global_step % 10 == 0:
#                             writer.add_scalar('test Dice score',val_score, global_step=global_step)
#                             writer.add_scalar('test IOU score',val_score2, global_step=global_step)
                  
#         if save_checkpoint:
#         #   if epoch > 200:
#            if epoch%2 == 0:
#             Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)

#             torch.save({
#             'epoch': epoch,
#             'model_state_dict': net1.module.state_dict(),
#             'optimizer_state_dict': optimizer.state_dict(),
#             'loss': loss,
#             }, str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch + 1)))

#             # torch.save(backbone.module.state_dict(), str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch + 1)))
#             logging.info(f'Checkpoint {epoch + 1} saved!')

# def setup_seed(seed):
#      torch.manual_seed(seed)
#      torch.cuda.manual_seed_all(seed)
#      np.random.seed(seed)
#      random.seed(seed)
#      torch.backends.cudnn.deterministic = True
# # 设置随机数种子


# if __name__ == '__main__':
#     #python -m torch.distributed.launch --nproc_per_node 2 main.py
#     args = parser.parse_args()
#     setup_seed(3407)
#     rank, word_size = setup_distributed(port=args.port)
#     # pretext_model = torch.load('/home/wjc20/segmentation/byol/baseline/train3/fusion51epoch37.pth')
#     pretext_model = torch.load('/home/wjc20/segmentation/byol/newidea/unet/pth/prefusion/checkpoint_epoch57.pth')
#     backbone = UNet(out_channels=2)
#     backbone = backbone.cuda()
#     backbone.load_state_dict(pretext_model['model_state_dict'],strict=False)
    
#     teacher = UNet(out_channels=2)
#     teacher = teacher.cuda()
#     teacher.load_state_dict(pretext_model['model_state_dict'],strict=False)
    
 
#     local_rank = int(os.environ["LOCAL_RANK"])
#     backbone = torch.nn.parallel.DistributedDataParallel(
#         backbone,
#         device_ids=[local_rank],
#         output_device=local_rank,
#         find_unused_parameters=False,
#     )
#     teacher = torch.nn.parallel.DistributedDataParallel(
#         teacher,
#         device_ids=[local_rank],
#         output_device=local_rank,
#         find_unused_parameters=False,
#     )
#     for p in teacher.parameters():
#         p.requires_grad = False
#     # for p in backbone.parameters():
#     #     p.requires_grad = False

#     logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
#     logging.info(f'Using device {device}')

    
#     try:
#         train_net(net1=backbone,
#                   net2=teacher,
#                   epochs=EPOCHS,
#                   batch_size=BATCH_SIZE,
#                   learning_rate=LR,
#                   device=device,
#                   val_percent=0.001
#                   )
#     except KeyboardInterrupt:
#         torch.save(backbone.module.state_dict(), '/home/wjc20/segmentation/byol/newidea/pth/epoch.pth') #2是864,1296预训练的batchsize4，  #1预训练的。1036-1555大小的 batch是3 70epoch
#         logging.info('Saved interrupt')