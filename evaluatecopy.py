import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '5,6'
# os.environ["OMP_NUM_THREADS"] = '32'
import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from dice_scorecopy import multiclass_dice_coeff, dice_coeff,multiclass_dice_coeff_val
import IPython
def evaluate(net, dataloader, device):
    net.eval()
    num_val_batches = len(dataloader)
  
    dice_score, IOU_score, Precision, Recall, Accuracy = 0,0,0,0,0
    valid_matrix = 0

    # iterate over the validation set
    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        image, label = batch[0], batch[1]
        label = label//200
        image = image.to(device=device, dtype=torch.float32)
        image = image/255
        label = label.to(device=device, dtype=torch.long)
        # label = torch.squeeze(label)
        # label = label[:,0,:]
       
        label = torch.squeeze(label)
        label = F.one_hot(label,2).permute(2, 0, 1).float()
        # label = F.one_hot(label, 2).permute(0, 3 ,1, 2).float() 

        with torch.no_grad():

            label_pred = net(image)
          
            label_pred = torch.squeeze(label_pred,0)
            label_pred = F.one_hot(label_pred.argmax(dim=0), 2).permute(2, 0, 1).float()
            
            dice_score1, IOU_score1, Precision1, Recall1, Accuracy1 = dice_coeff(label_pred[1, ...], label[1, ...],
                                                    reduce_batch_first=False)
            dice_score = dice_score + dice_score1
            IOU_score = IOU_score + IOU_score1
            Precision = Precision + Precision1
            Recall = Recall + Recall1
            Accuracy = Accuracy + Accuracy1
            
            # dice_score += dice_coeff(label_pred[1, ...], label[1, ...],
            #                                         reduce_batch_first=False)[0]
            # IOU_score += dice_coeff(label_pred[1, ...], label[1, ...],
            #                                         reduce_batch_first=False)[1]                                      


    net.train()
    # Fixes a potential division by zero error
    if num_val_batches == 0:
        return dice_score
    return (dice_score / num_val_batches, IOU_score / num_val_batches, 
            Precision / num_val_batches, Recall / num_val_batches, Accuracy / num_val_batches)
def u2plevaluate(net, dataloader, device):
    net.eval()
    num_val_batches = len(dataloader)
  
    dice_score, IOU_score, Precision, Recall, Accuracy = 0,0,0,0,0
    valid_matrix = 0

    # iterate over the validation set
    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        image, label = batch[0], batch[1]
        label = label//200
        image = image.to(device=device, dtype=torch.float32)
        image = torch.squeeze(image,dim=1)
        label = label.to(device=device, dtype=torch.long)
        # label = torch.squeeze(label)
        # label = label[:,0,:]
        
        # label = torch.squeeze(label,dim=1)
        label = torch.squeeze(label)#,dim=1)
        label = F.one_hot(label,2).permute(2, 0, 1).float()
        # label = F.one_hot(label, 2).permute(0, 3 ,1, 2).float() 

        with torch.no_grad():

            label_pred = net(image)
            # label_pred = label_pred[0]
            
           
            label_pred = torch.squeeze(label_pred,0)
            label_pred = F.one_hot(label_pred.argmax(dim=0), 2).permute(2, 0, 1).float()
           
            
            dice_score1, IOU_score1, Precision1, Recall1, Accuracy1 = dice_coeff(label_pred[1, ...], label[1, ...],
                                                    reduce_batch_first=False)
            dice_score = dice_score + dice_score1
            IOU_score = IOU_score + IOU_score1
            Precision = Precision + Precision1
            Recall = Recall + Recall1
            Accuracy = Accuracy + Accuracy1

    net.train()
    # Fixes a potential division by zero error
    if num_val_batches == 0:
        return dice_score
    return (dice_score / num_val_batches, IOU_score / num_val_batches, 
            Precision / num_val_batches, Recall / num_val_batches, Accuracy / num_val_batches)
def u2plevaluateval(net, dataloader, device):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score, IOU_score, Precision, Recall, Accuracy = 0,0,0,0,0
    valid_matrix = 0

    # iterate over the validation set
    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        image, label = batch[0], batch[1]
        label = label//200
        image = image.to(device=device, dtype=torch.float32)
        image = torch.squeeze(image,dim=1)
        label = label.to(device=device, dtype=torch.long)
        # label = torch.squeeze(label)
        # label = label[:,0,:]
       
        # label = torch.squeeze(label,dim=1)
        label = torch.squeeze(label)#,dim=1)
        label[label==2]=0
        
        label = F.one_hot(label,2).permute(2, 0, 1).float()
        # label = F.one_hot(label, 2).permute(0, 3 ,1, 2).float() 

        with torch.no_grad():

            label_pred = net(image)
            #unet+++有5个输出
            # label_pred = label_pred[0]
            label_pred = torch.squeeze(label_pred,0)
            label_pred = F.one_hot(label_pred.argmax(dim=0), 2).permute(2, 0, 1).float()
        
            
            dice_score1, IOU_score1, Precision1, Recall1, Accuracy1 = dice_coeff(label_pred[1, ...], label[1, ...],
                                                    reduce_batch_first=False)
            dice_score = dice_score + dice_score1
            IOU_score = IOU_score + IOU_score1
            Precision = Precision + Precision1
            Recall = Recall + Recall1
            Accuracy = Accuracy + Accuracy1

    net.train()
    # Fixes a potential division by zero error
    if num_val_batches == 0:
        return dice_score
    return (dice_score / num_val_batches, IOU_score / num_val_batches, 
            Precision / num_val_batches, Recall / num_val_batches, Accuracy / num_val_batches)

def trainevaluate(net,device,labels,labels_pred,batchsize):
    net.eval()
    num_val_batches = batchsize
    dice_score = 0
    IOU_score = 0  
    valid_matrix = 0
    label_preds, labels = labels_pred,labels

    with torch.no_grad():
            # prototypes,IScam,label_pred,proj = net(image)
     
        for i in range(batchsize):
            
            label_pred = label_preds[i]
            label = labels[i]
            
            
            dice_score += dice_coeff(label_pred[1, ...], label[1, ...],
                                                    reduce_batch_first=False)[0]
            IOU_score += dice_coeff(label_pred[1, ...], label[1, ...],
                                                    reduce_batch_first=False)[1]                                      


    net.train()
    # Fixes a potential division by zero error
    if num_val_batches == 0:
        return dice_score
    return (dice_score / num_val_batches, IOU_score / num_val_batches)
# def u2pltrain(net, dataloader, device,batchsize):
#     net.eval()
#     num_val_batches = len(dataloader)
#     dice_score = 0
#     IOU_score = 0  
#     valid_matrix = 0

#     # iterate over the validation set
#     for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
#         image, label = batch[0], batch[1]
#         label = label//200
#         image = image.to(device=device, dtype=torch.float32)
#         image = torch.squeeze(image,dim=1)
#         label = label.to(device=device, dtype=torch.long)
#         # label = torch.squeeze(label)
#         # label = label[:,0,:]
        
#         # label = torch.squeeze(label,dim=1)
#         label = torch.squeeze(label)#,dim=1)
#         label[label==2]=0
        
#         label = F.one_hot(label,2).permute(0,3, 1,2).float()
#         # label = F.one_hot(label, 2).permute(0, 3 ,1, 2).float() 

#         with torch.no_grad():
#           for i in range(batchsize):

#             label_pred = net(image)
            
           
#             label_pred = torch.squeeze(label_pred,0)
            
#             label_pred = F.one_hot(label_pred.argmax(dim=1), 2).permute(0,3, 1,2).float()
            
            
#             dice_score += dice_coeff(label_pred[i,1, ...], label[i,1, ...],
#                                                     reduce_batch_first=False)[0]
#             IOU_score += dice_coeff(label_pred[i,1, ...], label[i,1, ...],
#                                                     reduce_batch_first=False)[1]                                      

  
#     net.train()
#     # Fixes a potential division by zero error
#     if num_val_batches == 0:
#         return dice_score
#     return (dice_score / (num_val_batches*batchsize), IOU_score / (num_val_batches*batchsize))
