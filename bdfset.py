import copy
import math
import os
import os.path
import random
from cv2 import INTER_NEAREST
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
import cv2
import IPython
import augmentation as psp_trsform
import yaml
import copy
cfa = yaml.load(open('/home/wjc20/segmentation/byol/newidea/config.yaml', "r"), Loader=yaml.Loader)

cfs = cfa['dataset']
cfg = cfs['train']
cfv = cfs['val']
mean, std = cfg["mean"], cfg["std"]
def build_valtransfrom(cfg):
    trs_f= []
    mean, std = cfg["mean"], cfg["std"]
    trs_f.append(psp_trsform.ToTensor())
    trs_f.append(psp_trsform.Normalize(mean=mean, std=std))
    return psp_trsform.Compose(trs_f)
def build_transfrom(cfg):
    trs_form = []
    mean, std = cfg["mean"], cfg["std"]
    trs_form.append(psp_trsform.ToTensor())
    trs_form.append(psp_trsform.Normalize(mean=mean, std=std))
    if cfg.get("resize", False):
        trs_form.append(psp_trsform.Resize(cfg["resize"]))
    if cfg.get("rand_resize", False):
        trs_form.append(psp_trsform.RandResize(cfg["rand_resize"]))
    if cfg.get("rand_rotation", False):
        rand_rotation = cfg["rand_rotation"]
        trs_form.append(
            psp_trsform.RandRotate(rand_rotation)
        )
    if cfg.get("GaussianBlur", False) and cfg["GaussianBlur"]:
        trs_form.append(psp_trsform.RandomGaussianBlur())
    if cfg.get("flip", False) and cfg.get("flip"):
        trs_form.append(psp_trsform.RandomHorizontalFlip())
    if cfg.get("crop", False):
        crop_size, crop_type = cfg["crop"]["size"], cfg["crop"]["type"]
        trs_form.append(
            psp_trsform.Crop(crop_size, crop_type=crop_type)
        )
    return psp_trsform.Compose(trs_form)
class UnlabelData(Dataset):
    def __init__(self):
        self.sample_list1 = list()
        f = open('/home/wjc20/segmentation/byol/newidea/txt/unsup.txt', 'r')
        # f = open('/home/wjc20/segmentation/byol/newidea/unet/txt/0.5unsup.txt', 'r')
        # f = open('/home/wjc20/segmentation/byol/newidea/txt/unlabel.txt', 'r')
        # f = open('/home/wjc20/segmentation/byol/newidea/txt/105fusionimg.txt', 'r')
        line1 = f.readlines()
        for line in line1:
            self.sample_list1.append(line.strip())
        f.close()
        self.transform = build_transfrom(cfg)
    def __len__(self):
        return (len(self.sample_list1))

    def __getitem__(self, index):
        item1= self.sample_list1[index]
 
        img1 = cv2.imread(item1)
        
        # img1= cv2.resize(img1, (384,576),interpolation=INTER_NEAREST)
        img1 = np.transpose(img1, (2, 0, 1))
        
        img2 = copy.deepcopy(img1)
        img1,img2 = self.transform(img1,img2)
        # n1 = item1.split('img/')
        # n1 = item1.split('UV/')
        # n_1 = n1[1].split('.')
        # name1 = n1[1]
        
        return (img1,img2)#1是彩色图像，2是标签图。
class MyData(Dataset):
    def __init__(self):
        self.sample_list1 = list()
        self.sample_list2 = list()
     
        f = open('/home/wjc20/segmentation/byol/newidea/txt/51fusionimg.txt', 'r')
        # f = open('/home/wjc20/segmentation/byol/newidea/unet/txt/reimg.txt', 'r')
        # f = open('/home/wjc20/segmentation/byol/newidea/unet/txt/16reimg.txt', 'r')
        line1 = f.readlines()
        for line in line1:
            self.sample_list1.append(line.strip())
        f.close()
        f = open('/home/wjc20/segmentation/byol/newidea/txt/51fusionlabel.txt', 'r')
        # f = open('/home/wjc20/segmentation/byol/newidea/unet/txt/16relabel.txt', 'r')
        # f = open('/home/wjc20/segmentation/byol/newidea/unet/txt/relabel.txt', 'r')
        line2 = f.readlines()
        for linen in line2:
            self.sample_list2.append(linen.strip())
        f.close()
        self.transform = build_transfrom(cfg)
        
    def __len__(self):
        return (len(self.sample_list1))

    def __getitem__(self, index):
        item1= self.sample_list1[index]
        item2= self.sample_list2[index]
        # item3= self.sample_list3[index]

        img1 = cv2.imread(item1)
        
        # img1= cv2.resize(img1, (384,576), interpolation=INTER_NEAREST)

        img1 = np.transpose(img1, (2, 0, 1))


        img2 = cv2.imread(item2, cv2.IMREAD_GRAYSCALE)
        # img2 = cv2.resize(img2, (384,576), interpolation=INTER_NEAREST)
        img2 = np.expand_dims(img2, axis=2)
        img2 = np.transpose(img2, (2, 0, 1))
       
        
        img1,img2 = self.transform(img1,img2)
   
        return (img1,img2)#1是彩色图像，2是标签图。
class Labeldata(Dataset):
    def __init__(self):
        self.sample_list1 = list()
        self.sample_list2 = list()
        self.sample_list3 = list()
        # f = open('/home/wjc20/segmentation/byol/newidea/txt/51fusionimg.txt', 'r')
        f = open('/home/wjc20/segmentation/byol/newidea/txt/105huaimg.txt', 'r')
        # f = open('/home/wjc20/segmentation/byol/newidea/unet/txt/trainimg.txt', 'r')
        # f = open('/home/wjc20/segmentation/byol/newidea/unet/txt/233img.txt', 'r')
        line1 = f.readlines()
        for line in line1:
            self.sample_list1.append(line.strip())
        f.close()
        # f = open('/home/wjc20/segmentation/byol/newidea/txt/51fusionlabel.txt', 'r')
        f = open('/home/wjc20/segmentation/byol/newidea/txt/105hualabel.txt', 'r')
        # f = open('/home/wjc20/segmentation/byol/newidea/unet/txt/trainlabel.txt', 'r')
        # f = open('/home/wjc20/segmentation/byol/newidea/unet/txt/233label.txt', 'r')
        line2 = f.readlines()
        for linen in line2:
            self.sample_list2.append(linen.strip())
        f.close()
        self.transform = build_transfrom(cfg)
        
    def __len__(self):
        return (len(self.sample_list1))

    def __getitem__(self, index):
        item1= self.sample_list1[index]
        item2= self.sample_list2[index]
        # item3= self.sample_list3[index]

        img1 = cv2.imread(item1)
        
        # img1= cv2.resize(img1, (384,576), interpolation=INTER_NEAREST)

        img1 = np.transpose(img1, (2, 0, 1))


        img2 = cv2.imread(item2, cv2.IMREAD_GRAYSCALE)
        # img2 = cv2.resize(img2, (384,576), interpolation=INTER_NEAREST)
        img2 = np.expand_dims(img2, axis=2)
        img2 = np.transpose(img2, (2, 0, 1))
       
        
        img1,img2 = self.transform(img1,img2)
        # n1 = item1.split('me_json/')
        # # n1 = item1.split('UV/')
        # n_1 = n1[1].split('_R')
        # name = n_1[0]
        # img = np.array(img1,dtype='uint8')
        # img = np.squeeze(img)
        # img = np.transpose(img,(1,2,0))
      

        return (img1,img2)#1是彩色图像，2是标签图。

class TestData(Dataset):
    def __init__(self):
        self.sample_list1 = list()
        self.sample_list2 = list()

        f = open('/home/wjc20/segmentation/byol/newidea/unet/txt/huafentestimg.txt', 'r')  #测试集
        line1 = f.readlines()
        for line in line1:
            self.sample_list1.append(line.strip())
        f.close()

        f = open('/home/wjc20/segmentation/byol/newidea/unet/txt/huafentestlabel.txt', 'r')
        line1 = f.readlines()
        for line in line1:
            self.sample_list2.append(line.strip())
        f.close()
        self.transform = build_valtransfrom(cfg)

    def __len__(self):
        return (len(self.sample_list1))

    def __getitem__(self, index):
        item1= self.sample_list1[index]
        
        item2= self.sample_list2[index]
  
        img1 = cv2.imread(item1)
        img2 = cv2.imread(item2, cv2.IMREAD_GRAYSCALE)

        h, w, c= img1.shape
        if h<w:
            img1 = np.rot90(img1, 1)
            img2 = np.rot90(img2, 1)
        high, wid, c = img1.shape
        img1= cv2.resize(img1, (384,576), interpolation=INTER_NEAREST)
        img1 = np.transpose(img1, (2, 0, 1))

        img2 = cv2.resize(img2, (384,576), interpolation=INTER_NEAREST)
        img2 = np.expand_dims(img2, axis=2)
   
        
        img2 = np.transpose(img2, (2, 0, 1))
        img1,img2 = self.transform(img1,img2)
        
        
        return (img1,img2)#1是彩色图像，2是标签图。
class TestData1(Dataset):
    def __init__(self):
        self.sample_list1 = list()
        self.sample_list2 = list()

        f = open('/home/wjc20/segmentation/byol/newidea/unet/txt/refineimg.txt', 'r')  #测试集
        # f = open('/home/wjc20/segmentation/byol/newidea/unet/txt/trainimg.txt', 'r')
        line1 = f.readlines()
        for line in line1:
            self.sample_list1.append(line.strip())
        f.close()
      
        f = open('/home/wjc20/segmentation/byol/newidea/unet/txt/refinelabel.txt', 'r')
        # f = open('/home/wjc20/segmentation/byol/newidea/unet/txt/trainlabel.txt', 'r')
        line2 = f.readlines()
        for line in line2:
            self.sample_list2.append(line.strip())
        f.close()
        self.transform = build_valtransfrom(cfg)

    def __len__(self):
        return (len(self.sample_list1))

    def __getitem__(self, index):
        item1= self.sample_list1[index]
        
        # print(index)
        item2= self.sample_list2[index]
        
        img1 = cv2.imread(item1)
        img2 = cv2.imread(item2, cv2.IMREAD_GRAYSCALE)

        h, w, c= img1.shape
        if h<w:
            img1 = np.rot90(img1, 1)
            img2 = np.rot90(img2, 1)
        high, wid, c = img1.shape
        img1= cv2.resize(img1, (384,576), interpolation=INTER_NEAREST)
        img1 = np.transpose(img1, (2, 0, 1))

        img2 = cv2.resize(img2, (384,576), interpolation=INTER_NEAREST)
        img2 = np.expand_dims(img2, axis=2)
      
        
        img2 = np.transpose(img2, (2, 0, 1))
        img1,img2 = self.transform(img1,img2)

        # return (img1,img2)


        n1 = item1.split('_RGB')
        name1 = n1[0]
  
        name2 = (name1.split('json/'))[1]
        return (img1,img2,name2)#1是彩色图像，2是标签图。


class Copypastedata(Dataset):
    def __init__(self):
        self.sample_list1 = list()
        self.sample_list2 = list()
        self.sample_list3 = list()
        # f = open('/home/wjc20/segmentation/byol/newidea/txt/51fusionimg.txt', 'r')
        f = open('/home/wjc20/segmentation/byol/newidea/unet/txt/copypasteimg.txt', 'r')
        # f = open('/home/wjc20/segmentation/byol/newidea/unet/txt/rehuafen.txt', 'r')
        line1 = f.readlines()
        for line in line1:
            self.sample_list1.append(line.strip())
        f.close()
        # f = open('/home/wjc20/segmentation/byol/newidea/txt/51fusionlabel.txt', 'r')
        f = open('/home/wjc20/segmentation/byol/newidea/unet/txt/copypastelabel.txt', 'r')
        # f = open('/home/wjc20/segmentation/byol/newidea/unet/txt/rehuafenlabel.txt', 'r')
        line2 = f.readlines()
        for linen in line2:
            self.sample_list2.append(linen.strip())
        f.close()
        self.transform = build_transfrom(cfg)
        
    def __len__(self):
        return (len(self.sample_list1))

    def __getitem__(self, index):
        item1= self.sample_list1[index]
        item2= self.sample_list2[index]
        # item3= self.sample_list3[index]

        img1 = cv2.imread(item1)
        
        # img1= cv2.resize(img1, (384,576), interpolation=INTER_NEAREST)

        img1 = np.transpose(img1, (2, 0, 1))


        img2 = cv2.imread(item2, cv2.IMREAD_GRAYSCALE)
        # img2 = cv2.resize(img2, (384,576), interpolation=INTER_NEAREST)
        img2 = np.expand_dims(img2, axis=2)
        img2 = np.transpose(img2, (2, 0, 1))
        img2[img2!=0] = 255
       
        
        img1,img2 = self.transform(img1,img2)
     
        return (img1,img2)#1是彩色图像，2是标签图。

class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img ,label):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(2)
        w = img.size(3)

        mask = np.ones((h, w), np.float32)
        mask2 = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.
            mask2[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)

        mask = mask.expand_as(img)
        
        
        mask2 = torch.from_numpy(mask2)
        mask2 = mask2.expand_as(label)
        # IPython.embed()
        img = img * mask
        label = label * mask2

        return img,label
def cutout_transfrom(cfg):
    trs_form = []
    cut = Cutout(n_holes=12, length=20)
    mean, std = cfg["mean"], cfg["std"]
    trs_form.append(psp_trsform.ToTensor())
    trs_form.append(psp_trsform.Normalize(mean=mean, std=std))
    if cfg.get("resize", False):
        trs_form.append(psp_trsform.Resize(cfg["resize"]))
    if cfg.get("rand_resize", False):
        trs_form.append(psp_trsform.RandResize(cfg["rand_resize"]))
    if cfg.get("rand_rotation", False):
        rand_rotation = cfg["rand_rotation"]
        trs_form.append(
            psp_trsform.RandRotate(rand_rotation)
        )
    if cfg.get("GaussianBlur", False) and cfg["GaussianBlur"]:
        trs_form.append(psp_trsform.RandomGaussianBlur())
    if cfg.get("flip", False) and cfg.get("flip"):
        trs_form.append(psp_trsform.RandomHorizontalFlip())
    if cfg.get("crop", False):
        crop_size, crop_type = cfg["crop"]["size"], cfg["crop"]["type"]
        trs_form.append(
            psp_trsform.Crop(crop_size, crop_type=crop_type)
        )
    trs_form.append(cut)
    return psp_trsform.Compose(trs_form)
class Cutoutdata(Dataset):
    def __init__(self):
        self.sample_list1 = list()
        self.sample_list2 = list()
        self.sample_list3 = list()
        # f = open('/home/wjc20/segmentation/byol/newidea/txt/51fusionimg.txt', 'r')
        # f = open('/home/wjc20/segmentation/byol/newidea/unet/txt/trainimg.txt', 'r')
        f = open('/home/wjc20/segmentation/byol/newidea/txt/105huaimg.txt', 'r')
        line1 = f.readlines()
        for line in line1:
            self.sample_list1.append(line.strip())
        f.close()
        # f = open('/home/wjc20/segmentation/byol/newidea/txt/51fusionlabel.txt', 'r')
        # f = open('/home/wjc20/segmentation/byol/newidea/unet/txt/trainlabel.txt', 'r')
        f = open('/home/wjc20/segmentation/byol/newidea/txt/105hualabel.txt', 'r')
        line2 = f.readlines()
        for linen in line2:
            self.sample_list2.append(linen.strip())
        f.close()
        self.transform = cutout_transfrom(cfg)
        
    def __len__(self):
        return (len(self.sample_list1))

    def __getitem__(self, index):
        item1= self.sample_list1[index]
        item2= self.sample_list2[index]
        # item3= self.sample_list3[index]

        img1 = cv2.imread(item1)
        
        # img1= cv2.resize(img1, (384,576), interpolation=INTER_NEAREST)
   
        img1 = np.transpose(img1, (2, 0, 1))


        img2 = cv2.imread(item2, cv2.IMREAD_GRAYSCALE)
        # img2 = cv2.resize(img2, (384,576), interpolation=INTER_NEAREST)
        img2 = np.expand_dims(img2, axis=2)
        img2 = np.transpose(img2, (2, 0, 1))
        img2[img2!=0] = 255
       
        
        img1,img2 = self.transform(img1,img2)
        n1 = item1.split('me_json/')
        # n1 = item1.split('UV/')
        # IPython.embed()
        # n_1 = n1[1].split('_R')
        # name = n_1[0]
        # img = np.array(img1,dtype='uint8')
        # img = np.squeeze(img)
        # img = np.transpose(img,(1,2,0))
        # cv2.imwrite('/home/wjc20/segmentation/byol/newidea/unet/conimg/aug/cutout/img{}.png'.format(name), img)

        return (img1,img2)#1是彩色图像，2是标签图。

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
#
class Grid(object):
    def __init__(self, d1, d2, rotate = 1, ratio = 0.5, mode=1, prob=1.):
        self.d1 = d1
        self.d2 = d2
        self.rotate = rotate
        self.ratio = ratio
        self.mode=mode
        self.st_prob = self.prob = prob

    def set_prob(self, epoch, max_epoch):
        self.prob = self.st_prob * min(1, epoch / max_epoch)

    def __call__(self, img, label):
        if np.random.rand() > self.prob:
            return img
        h = img.size(2)
        w = img.size(3)
        
        # 1.5 * h, 1.5 * w works fine with the squared images
        # But with rectangular input, the mask might not be able to recover back to the input image shape
        # A square mask with edge length equal to the diagnoal of the input image 
        # will be able to cover all the image spot after the rotation. This is also the minimum square.
        hh = math.ceil((math.sqrt(h*h + w*w)))
        
        d = np.random.randint(self.d1, self.d2)
        #d = self.d
        
        # maybe use ceil? but i guess no big difference
        self.l = math.ceil(d*self.ratio)
        
        mask = np.ones((hh, hh), np.float32)
        st_h = np.random.randint(d)
        st_w = np.random.randint(d)
        for i in range(-1, hh//d+1):
                s = d*i + st_h
                t = s+self.l
                s = max(min(s, hh), 0)
                t = max(min(t, hh), 0)
                mask[s:t,:] *= 0
        for i in range(-1, hh//d+1):
                s = d*i + st_w
                t = s+self.l
                s = max(min(s, hh), 0)
                t = max(min(t, hh), 0)
                mask[:,s:t] *= 0
        r = np.random.randint(self.rotate)
        mask = Image.fromarray(np.uint8(mask))
        mask = mask.rotate(r)
        mask = np.asarray(mask)
        mask = mask[(hh-h)//2:(hh-h)//2+h, (hh-w)//2:(hh-w)//2+w]

        mask = torch.from_numpy(mask).float().cuda()
        if self.mode == 1:
            mask = 1-mask
        # IPython.embed()
        

        # mask = mask.expand_as(img)
        # mask = mask.cpu().numpy()
        mask = mask.unsqueeze(dim=0)
        mask = mask.unsqueeze(dim=0)
        # mask = mask[np.newaxis,np.newaxis,:]
        mask = mask.cpu().numpy()
        label = label* mask
        img = img * mask 
        # IPython.embed()
        # label = label* mask

        return img,label

class GridMask(nn.Module):
    def __init__(self, d1, d2, rotate = 1, ratio = 0.25, mode=0, prob=1.):
        super(GridMask, self).__init__()
        self.rotate = rotate
        self.ratio = ratio
        self.mode = mode
        self.st_prob = prob
        self.grid = Grid(d1, d2, rotate, ratio, mode, prob)

    def set_prob(self, epoch, max_epoch):
        self.grid.set_prob(epoch, max_epoch)

    def forward(self, x):
        if not self.training:
            return x
        n,c,h,w = x.size()
        y = []
        for i in range(n):
            y.append(self.grid(x[i]))
        y = torch.cat(y).view(n,c,h,w)
        return y


def gridmask_transfrom(cfg):
    trs_form = []
    cut = Grid(d1=40,d2=60)
    mean, std = cfg["mean"], cfg["std"]
    trs_form.append(psp_trsform.ToTensor())
    trs_form.append(psp_trsform.Normalize(mean=mean, std=std))
    if cfg.get("resize", False):
        trs_form.append(psp_trsform.Resize(cfg["resize"]))
    if cfg.get("rand_resize", False):
        trs_form.append(psp_trsform.RandResize(cfg["rand_resize"]))
    if cfg.get("rand_rotation", False):
        rand_rotation = cfg["rand_rotation"]
        trs_form.append(
            psp_trsform.RandRotate(rand_rotation)
        )
    if cfg.get("GaussianBlur", False) and cfg["GaussianBlur"]:
        trs_form.append(psp_trsform.RandomGaussianBlur())
    if cfg.get("flip", False) and cfg.get("flip"):
        trs_form.append(psp_trsform.RandomHorizontalFlip())
    if cfg.get("crop", False):
        crop_size, crop_type = cfg["crop"]["size"], cfg["crop"]["type"]
        trs_form.append(
            psp_trsform.Crop(crop_size, crop_type=crop_type)
        )
    trs_form.append(cut)
    return psp_trsform.Compose(trs_form)
class gridmaskdata(Dataset):
    def __init__(self):
        self.sample_list1 = list()
        self.sample_list2 = list()
        self.sample_list3 = list()
        # f = open('/home/wjc20/segmentation/byol/newidea/txt/51fusionimg.txt', 'r')
        # f = open('/home/wjc20/segmentation/byol/newidea/unet/txt/trainimg.txt', 'r')
        f = open('/home/wjc20/segmentation/byol/newidea/txt/105huaimg.txt', 'r')
        line1 = f.readlines()
        for line in line1:
            self.sample_list1.append(line.strip())
        f.close()
        # f = open('/home/wjc20/segmentation/byol/newidea/txt/51fusionlabel.txt', 'r')
        # f = open('/home/wjc20/segmentation/byol/newidea/unet/txt/trainlabel.txt', 'r')
        f = open('/home/wjc20/segmentation/byol/newidea/txt/105hualabel.txt', 'r')
        line2 = f.readlines()
        for linen in line2:
            self.sample_list2.append(linen.strip())
        f.close()
        self.transform = gridmask_transfrom(cfg)
        
    def __len__(self):
        return (len(self.sample_list1))

    def __getitem__(self, index):
        item1= self.sample_list1[index]
        item2= self.sample_list2[index]
        # item3= self.sample_list3[index]

        img1 = cv2.imread(item1)
        
        # img1= cv2.resize(img1, (384,576), interpolation=INTER_NEAREST)

        img1 = np.transpose(img1, (2, 0, 1))


        img2 = cv2.imread(item2, cv2.IMREAD_GRAYSCALE)
        # img2 = cv2.resize(img2, (384,576), interpolation=INTER_NEAREST)
        img2 = np.expand_dims(img2, axis=2)
        img2 = np.transpose(img2, (2, 0, 1))
        img2[img2!=0] = 255
       
    
        # img1 = np.squeeze(img1)
        # img2 = np.squeeze(img2, axis=0)
    
        img1,img2 = self.transform(img1,img2)
        # n1 = item1.split('me_json/')
        # n1 = item1.split('UV/')
        # n_1 = n1[1].split('_R')
        # name = n_1[0]
        # img = np.array(img1,dtype='uint8')
        # img = np.squeeze(img)
        # img = np.transpose(img,(1,2,0))
        # cv2.imwrite('/home/wjc20/segmentation/byol/newidea/unet/conimg/aug/gridmask1/img{}.png'.format(name), img)
           
        return (img1,img2)#1是彩色图像，2是标签图。
