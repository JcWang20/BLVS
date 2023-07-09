import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '3,'
# os.environ["OMP_NUM_THREADS"] = '32'
import torch
from torch import Tensor
import copy
import IPython
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-8):
    # To Do:  cross-validation of accuracy, precision, recall, and F1-score for each fold
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    if input.dim() == 2 and reduce_batch_first:
        raise ValueError(f'Dice: asked to reduce batch but got tensor without batch dimension (shape {input.shape})')
    # import IPython
    # IPython.embed()

    if input.dim() == 2  or reduce_batch_first:

        inter = torch.dot(torch.flatten(input), torch.flatten(target)) #TP
        # inputcopy = copy.deepcopy(input)
        
        # targetcopy = copy.deepcopy(target)
        
        
        aa = input*target
        TP = torch.sum(aa>0.5)
        positive = torch.sum(target)
        # IPython.embed() 
        Total = torch.tensor(221184)
        negative = Total - positive
        FP = torch.sum((input*(1-target))>0.5)  #FP
        TN = negative - FP
        FN = positive - TP
        # TN = Total - (TP.sum() + FP.sum() + FN.sum()) 
        # IOU = (TP.sum()+ epsilon) / ((TP.sum() + FP.sum() + FN.sum())+epsilon)
        # IOU = TP.sum() / (input.sum() + FN.sum()-TP.sum())

        sets_sum = torch.sum(input) + torch.sum(target) 
        
        union = sets_sum - inter
        IOU = inter / (union + epsilon)
        # IPython.embed()
        Precision = torch.true_divide(TP,(TP+ FP))
        Recall = torch.true_divide(TP, (TP + FN))
        Accuracy = torch.true_divide((TP + TN),Total)
        # IPython.embed()
        if sets_sum.item() == 0:
            sets_sum = 2 * inter
        
        
        return ((2 * inter) / (sets_sum + epsilon),IOU,Precision,Recall,Accuracy)
    else:
        # compute and average metric for each batch element
        dice1 = 0
        IOU1 = 0
        Precision1,Recall1,Accuracy1 = 0,0,0
        #pring('aaaaaa')
        # IPython.embed()
        for i in range(input.shape[0]):
            dice, IOU, Precision, Recall, Accuracy = dice_coeff(input[i, ...], target[i, ...])
            dice1 = dice + dice1
            IOU1  = IOU + IOU1
            Precision1 = Precision +Precision1
            Recall1 = Recall + Recall1
            Accuracy1 = Accuracy + Accuracy1
           
            # dice += dice_coeff(input[i, ...], target[i, ...])[0]
            # IOU += dice_coeff(input[i, ...], target[i, ...])[1]
            # Precision += dice_coeff(input[i, ...], target[i, ...])[2]
            # Recall += dice_coeff(input[i, ...], target[i, ...])[3]
            # Accuracy += dice_coeff(input[i, ...], target[i, ...])[4]
        # IPython.embed()
        return (dice1/input.shape[0], IOU1/input.shape[0], Precision1/input.shape[0], Recall1/input.shape[0], Accuracy1/input.shape[0])
        # return (dice, IOU)
def dice_coeffvit(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-8):
    # To Do:  cross-validation of accuracy, precision, recall, and F1-score for each fold
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    if input.dim() == 2 and reduce_batch_first:
        raise ValueError(f'Dice: asked to reduce batch but got tensor without batch dimension (shape {input.shape})')
    # import IPython
    # IPython.embed()

    if input.dim() == 2  or reduce_batch_first:

        inter = torch.dot(torch.flatten(input), torch.flatten(target)) #TP
        # inputcopy = copy.deepcopy(input)
        
        # targetcopy = copy.deepcopy(target)
        
        
        aa = input*target
        TP = torch.sum(aa>0.5)
        positive = torch.sum(target)
        # IPython.embed() 
        Total = torch.tensor(221184)
        negative = Total - positive
        FP = torch.sum((input*(1-target))>0.5)  #FP
        TN = negative - FP
        FN = positive - TP
        # TN = Total - (TP.sum() + FP.sum() + FN.sum()) 
        # IOU = (TP.sum()+ epsilon) / ((TP.sum() + FP.sum() + FN.sum())+epsilon)
        # IOU = TP.sum() / (input.sum() + FN.sum()-TP.sum())

        sets_sum = torch.sum(input) + torch.sum(target) 
        
        union = sets_sum - inter
        IOU = inter / (union + epsilon)
        Precision = torch.true_divide(TP,(TP+ FP+epsilon))
        Recall = torch.true_divide(TP, (TP + FN+epsilon))
        Accuracy = torch.true_divide((TP + TN),Total)
        # IPython.embed()
        # IPython.embed()  epsilon=1e-8
        if sets_sum.item() == 0: 
            sets_sum = 2 * inter
        
        
        return ((2 * inter) / (sets_sum + epsilon),IOU,Precision,Recall,Accuracy)
    else:
        # compute and average metric for each batch element
        dice1 = 0
        IOU1 = 0
        Precision1,Recall1,Accuracy1 = 0,0,0
        #pring('aaaaaa')
        # IPython.embed()
        for i in range(input.shape[0]):
            dice, IOU, Precision, Recall, Accuracy = dice_coeff(input[i, ...], target[i, ...])
            dice1 = dice + dice1
            IOU1  = IOU + IOU1
            Precision1 = Precision +Precision1
            Recall1 = Recall + Recall1
            Accuracy1 = Accuracy + Accuracy1
           
            # dice += dice_coeff(input[i, ...], target[i, ...])[0]
            # IOU += dice_coeff(input[i, ...], target[i, ...])[1]
            # Precision += dice_coeff(input[i, ...], target[i, ...])[2]
            # Recall += dice_coeff(input[i, ...], target[i, ...])[3]
            # Accuracy += dice_coeff(input[i, ...], target[i, ...])[4]
        # IPython.embed()
        return (dice1/input.shape[0], IOU1/input.shape[0], Precision1/input.shape[0], Recall1/input.shape[0], Accuracy1/input.shape[0])
def dice_coeffloss(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-8):
    # To Do:  cross-validation of accuracy, precision, recall, and F1-score for each fold
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    if input.dim() == 2 and reduce_batch_first:
        raise ValueError(f'Dice: asked to reduce batch but got tensor without batch dimension (shape {input.shape})')
    # import IPython
    # IPython.embed()

    if input.dim() == 2  or reduce_batch_first:

        inter = torch.dot(torch.flatten(input), torch.flatten(target)) #TP
        sets_sum = torch.sum(input) + torch.sum(target) 
        


        if sets_sum.item() == 0:
            sets_sum = 2 * inter
        
        
        return ((2 * inter) / (sets_sum + epsilon))
    else:
        # compute and average metric for each batch element
        dice1 = 0
    
        #pring('aaaaaa')
        # IPython.embed()
        for i in range(input.shape[0]):
            dice = dice_coeffloss(input[i, ...], target[i, ...])
            dice1 = dice + dice1
           
        return (dice1/input.shape[0])
        # return (dice, IOU)



def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all classes
    assert input.size() == target.size()
    dice = 0
    for channel in range(input.shape[1]):
        dice += dice_coeff(input[:, channel, ...], target[:, channel, ...], reduce_batch_first, epsilon)[0]
        iou += dice_coeff(input[:, channel, ...], target[:, channel, ...], reduce_batch_first, epsilon)[1]
    #print(input.shape)
    #print(input)
    return (dice / (input.shape[1]) ,iou / (input.shape[1]))

def multiclass_dice_coeff_val(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all classes
    assert input.size() == target.size()
    dice = 0
    for channel in range(input.shape[1]):
        dice += dice_coeff(input[:, channel, ...], target[:, channel, ...], reduce_batch_first, epsilon)[0]
        iou += dice_coeff(input[:, channel, ...], target[:, channel, ...], reduce_batch_first, epsilon)[1]

    return (dice / (input.shape[1]) ,iou / (input.shape[1]))


def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    assert input.size() == target.size()
    # mask掉第21类背景类，计算loss的时候不管这个类
    fn = multiclass_dice_coeff if multiclass else dice_coeffloss
    
    return 1 - fn(input, target, reduce_batch_first=False)