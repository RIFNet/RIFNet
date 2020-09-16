# -*- coding: utf-8 -*-
import random
import os
from PIL import Image,ImageFilter,ImageDraw
import numpy as np
import h5py
from PIL import ImageStat
import cv2
import scipy.io as io

def load_data(img_path,train = True):
    gt_path = img_path.replace('.png','.h5').replace('pngs','ground_truth_v=3.5')
    img = Image.open(img_path).convert('RGB')
    img = np.asarray(img)
    
    gt_file = h5py.File(gt_path,'r')
    target = np.asarray(gt_file['density'])
    
    s_path = img_path.replace('.png','.h5').replace('pngs','signal_density_real')
    s_file = h5py.File(s_path,'r')
    signal = np.asarray(s_file['density'])
    signal_down = cv2.resize(signal,(int(signal.shape[1]/8),int(signal.shape[0]/8)),interpolation = cv2.INTER_CUBIC)*64
    
    mat = io.loadmat(img_path.replace('.png','.mat').replace('pngs','mats'))
    signal = np.zeros((225,400))
    gt = mat["image_info"]
    for i in range(0,len(gt)):
        if int(gt[i][0])<img.shape[0] and int(gt[i][1])<img.shape[1]:
            signal[int(gt[i][0]),int(gt[i][1])]=1
    
    target = cv2.resize(target,(int(target.shape[1]),int(target.shape[0])),interpolation = cv2.INTER_CUBIC)
    
    return img,target,signal,signal_down
