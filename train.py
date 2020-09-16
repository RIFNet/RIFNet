import sys
import os

import warnings

from model import RIFNet

from utils import save_checkpoint

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import argparse
import json
import cv2
import dataset
import time
import glob
import h5py
import math
import random

parser = argparse.ArgumentParser(description='PyTorch RIFNet')
writer = SummaryWriter()

parser.add_argument('--pre', '-p', metavar='PRETRAINED', default=None,type=str,
                    help='path to the pretrained model')

parser.add_argument('gpu',metavar='GPU', type=str,
                    help='GPU id to use.')

parser.add_argument('task',metavar='TASK', type=str,
                    help='task id to use.')

# +
def main():
    
    global args,best_prec1
    
    best_prec1 = 1e6
    best_MPAE = 1e6
    
    args = parser.parse_args()
    args.original_lr = 1e-4
    args.lr = 1e-6
    args.batch_size    = 1
    args.momentum      = 0.95
    args.decay         = 5*1e-4
    args.start_epoch   = 0
    args.epochs = 400
    args.steps         = [40,50,60,70,80]
    args.scales        = [1,1,1,1]
    args.workers = 4
    args.seed = time.time()
    args.print_freq = 30
    
    #load dataset
    train_list = []
    for img_path in glob.glob('../UAV_GCC_all/train/*/pngs/*.png'):
        train_list.append(img_path)
    
    val_list = []
    for img_path in glob.glob('../UAV_GCC_all/val/*/pngs/*.png'):
        val_list.append(img_path)
    
    #read ring_map and ring_weight
    ring = np.zeros((24,225,400),dtype=float)
    for i in range(24):
        ring_ = h5py.File('../UAV_GCC_all/ring/'+str(i)+'.h5','r')
        ring[i] = np.asarray(ring_['ring'])
    
    ring_= h5py.File('../UAV_GCC_all/ring_weight/ring_weight.h5','r')
    ring_weight = np.asarray(ring_['ring_weight'])
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    torch.cuda.manual_seed(args.seed)
    
    model = RIFNet()
    
    model = model.cuda()
    
#     #freeze weight
#     t=0
#     for s in model.backend_1:
#         if(t%2==0):
#             s.weight.requires_grad = False
#             s.bias.requires_grad = False
#         t+=1
#     t=0
#     for s in model.backend_2:
#         if(t%2==0):
#             s.weight.requires_grad = False
#             s.bias.requires_grad = False
#         t+=1
    
#     model.output_layer.weight.requires_grad = False
#     model.output_layer.bias.requires_grad = False
    
    criterion = nn.MSELoss(size_average=False).cuda()
    
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), args.lr, weight_decay=args.decay)

    
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)
        
        train(train_list, model, criterion, optimizer, epoch,ring,ring_weight)
        prec1,s,MPAE= validate(val_list, model, criterion)
        
        writer.add_scalar('PMAE',prec1,epoch)
        writer.add_scalar('MAE',s,epoch)

        is_best = prec1 < best_prec1
        is_best_MPAE = MPAE < best_MPAE
        best_prec1 = min(prec1, best_prec1)
        best_MPAE = min(MPAE, best_MPAE)
        print(' * best PMAE {pmae:.4f} '
              .format(pmae=best_prec1))
        print(' * best MPAE {mpae:.4f} '
              .format(mpae=best_MPAE))
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.pre,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, is_best,args.task)

# +
def train(train_list, model, criterion, optimizer, epoch,ring,ring_weight):
    
    ring = torch.from_numpy(ring).type(torch.FloatTensor).cuda()
    ring_weight = torch.from_numpy(ring_weight).type(torch.FloatTensor).cuda()
    ring = Variable(ring)
    ring_weight = Variable(ring_weight)
    
    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    
    train_loader = torch.utils.data.DataLoader(
        dataset.listDataset(train_list,
                       shuffle=True,
                       transform=transforms.Compose([
                       transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
                   ]), 
                       train=True, 
                       seen=model.seen,
                       batch_size=args.batch_size,
                       num_workers=args.workers),
        batch_size=args.batch_size)
    print('epoch %d, processed %d samples, lr %.10f' % (epoch, epoch * len(train_loader.dataset), args.lr))
    
    model.train()
    end = time.time()

    for i,(img, target,signal,signal_down)in enumerate(train_loader):
        data_time.update(time.time() - end)

        img = img.cuda()
        img = Variable(img)
        sig_down = signal_down.type(torch.FloatTensor).unsqueeze(0).cuda()
        sig_down = Variable(sig_down)
        sig = signal.type(torch.FloatTensor).unsqueeze(0).cuda()
        sig = Variable(sig)
    
    #compute round people
        round_count = torch.zeros([24]).type(torch.FloatTensor).cuda()
        round_count = Variable(round_count)
        for k in range(24):
            round_count[k] = torch.mul(sig,ring[k]).sum()
        sig = round_count
        
#     #random generate a mask that mask out a ring
#         rand = random.randint(0,174)
#         rand2 = random.randint(0,349)
#         for q in range(3):
#             img[:,q,rand:rand+50,rand2:rand2+50] = 0 

        output = model(img,sig_down)
        
        target = target.type(torch.FloatTensor).unsqueeze(0).cuda()
        target = Variable(target)
        
        ##multiply ring weight
        total_sum = torch.tensor([0]).type(torch.FloatTensor).cuda()
        total_sum = Variable(total_sum)
        for k in range(24):
            ring_sum = torch.abs(torch.mul(output,ring[k]).sum()-torch.mul(target,ring[k]).sum())
            total_sum += ring_sum*ring_weight[k]
        
        loss = criterion(output,target)+ 0.005 * total_sum

        losses.update(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time() 
        
        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  .format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))
# -

def validate(val_list, model, criterion):
    print ('begin test')
    test_loader = torch.utils.data.DataLoader(
    dataset.listDataset(val_list,
                   shuffle=False,
                   transform=transforms.Compose([
                       transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
                   ]),  train=False),
    batch_size=1)

    model.eval()

    mae = 0
    MPAE = 0
    PMAE = 0
    for i,(img, target,signal,signal_down) in enumerate(test_loader):
        img = img.cuda()
        img = Variable(img)
        
        sig_down = signal_down.type(torch.FloatTensor).unsqueeze(0).cuda()
        sig_down = Variable(sig_down)
        sig = signal.type(torch.FloatTensor).unsqueeze(0).cuda()
        sig = Variable(sig)

        output = model(img,sig_down)
        target = target.type(torch.FloatTensor).unsqueeze(0).cuda()
        target = Variable(target,requires_grad=False)
        mae += abs(output.data.sum()-target.sum())
        
        MPAE += abs(output.squeeze(0).squeeze(0).data - target.squeeze(0).squeeze(0).data).sum()
        
        #patch size 25x25  #total 16*9 patches
        temp = 0
        for m in range(0,225,75):
            for n in range(0,400,100):
                temp += abs(output.squeeze(0).squeeze(0)[m:m+74,n:n+99].data.sum()-target.squeeze(0).squeeze(0)[m:m+74,n:n+99].data.sum())
        PMAE += temp
    MPAE = MPAE.cpu()
    
    mae = mae/len(test_loader)
    MPAE = MPAE/len(test_loader)
    PMAE = PMAE/len(test_loader)/4/3
    print(' * PMAE {pmae:.4f} '
              .format(pmae=PMAE))
    print(' * MPAE {mpae:.4f} '
              .format(mpae=MPAE))
    
    return PMAE,mae,MPAE

def adjust_learning_rate(optimizer, epoch):
    args.lr = args.original_lr
    
    for i in range(len(args.steps)):
        
        scale = args.scales[i] if i < len(args.scales) else 1
        
        
        if epoch >= args.steps[i]:
            args.lr = args.lr * scale
            if epoch == args.steps[i]:
                break
        else:
            break
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.lr

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

if __name__ == '__main__':
    main()        
