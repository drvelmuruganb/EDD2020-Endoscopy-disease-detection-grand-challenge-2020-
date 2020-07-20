#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 00:27:04 2018

@author: sumanthnandamuri
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import tqdm
import time
import gc

from SUMNet_sig import SUMNet
from utils import dice_coefficient, make_loader
from edd_loader import eddLoader
from torch.utils import data
from torchvision import transforms

# from aug_utils import *
import augmentation as aug
import sys
import os
from PIL import Image
import cv2

from scipy import ndimage
import matplotlib.pyplot as plt

# import scipy.misc as sm

import tifffile as tiff


savePath = 'Results/SumNet_with_augs/'
M = 256 #image size
outpath = savePath
net = SUMNet()
# checkpoint = torch.load(weight_path)
net.load_state_dict(torch.load(savePath+'SumNet_with_augs/SUMNet_class0_best.pt'))#saved weight path
net.eval()
use_gpu = torch.cuda.is_available()
if use_gpu:
    net = net.cuda()


tf = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

net.train(False)

loadPath = '/home/histosr/Desktop/Media/EDD_evaluation_test-Final/'
test_list = os.listdir(loadPath)


bbox_folder = savePath+'EDD_evaluation_test-Final_lrrc1/EndoCV2020_testSubmission/detection_bbox/'
if not os.path.isdir(bbox_folder):
    os.makedirs(bbox_folder)

img_folder = savePath+'EDD_evaluation_test-Final_lrrc1/EndoCV2020_testSubmission/semantic_masks/'
if not os.path.isdir(img_folder):
    os.makedirs(img_folder)

# img_folder_256 = savePath+'EDD_evaluation_test-Final_bestPolyp_thresh0.5_CC_bestBE/masks_256/'
# if not os.path.isdir(img_folder_256):
#     os.makedirs(img_folder_256)

# def CC(Map):
#     label_img, cc_num = ndimage.label(Map)
#     CC = ndimage.find_objects(label_img)
#     cc_areas = ndimage.sum(Map, label_img, range(cc_num+1))
#     area_mask = (cc_areas < 1000000)
#     label_img[area_mask[label_img]] = 0
#     return label_img, CC

class_names = ['BE','suspicious','HGD','cancer','polyp']

for test_img_name in test_list:
    

    if (test_img_name == 'EDD2020_test_21.jpg') | (test_img_name == 'EDD2020_test_25.jpg'):
    # if test_img_name == 'EDD2020_test_21.jpg':
        print(test_img_name)


        detecttion_list = []

        test_img = Image.open(loadPath+test_img_name)
        img_orig_size = test_img.size
        test_img = test_img.resize((M,M))

        test_inp = tf(test_img)

            
        if use_gpu:
            test_inp = test_inp.cuda()


        ############################### For network with sigmoid ################
        probs = net(test_inp.unsqueeze(0)).squeeze(0).cpu().detach()    
        # print(probs.max())

        thresh = 0.5*probs.max()
        preds = (probs > thresh).float()

        preds[preds>0] = 255
        pred_np = np.asarray(preds.numpy(),dtype=np.uint8)
        

        ####### Resize to original size #################
        pred_data = np.zeros((5,img_orig_size[1],img_orig_size[0]))
        probs_data = np.zeros((img_orig_size[1],img_orig_size[0]))

        mask_save = np.zeros((5,img_orig_size[1],img_orig_size[0]),dtype=np.uint8)

        for classNum in range(5):
            pred_data[classNum] = np.uint8(np.array(Image.fromarray(pred_np[classNum]).resize(img_orig_size)))
            probs_data = np.array(Image.fromarray(probs.numpy()[classNum]).resize(img_orig_size))

            outlier_idx = np.where(probs_data>1)
            probs_data[outlier_idx[0],outlier_idx[1]] = probs_data[outlier_idx[0],outlier_idx[1]]/255.0


            pred_binary = np.uint8(pred_data[classNum].copy())

            pred_binary[pred_binary>0] = 1    
                    
                
            output = cv2.connectedComponentsWithStats(pred_binary, 4, cv2.CV_32S)        
            labels = output[1]        
            bbox_stats = output[2]

            
            for compNum in range(1,len(bbox_stats)):
                x_1 = bbox_stats[compNum][0] #Horizontal
                y_1 = bbox_stats[compNum][1] #Vertical
                x_2 = x_1 + bbox_stats[compNum][2]
                y_2 = y_1 + bbox_stats[compNum][3]

                area = bbox_stats[compNum][4]

                if area>100:                    
                    label_idx = np.where(labels==compNum)
                    mask_save[classNum,label_idx[0],label_idx[1]] = 255            
                    conf_score = probs_data[label_idx[0],label_idx[1]].mean()
                    detecttion_list.append([class_names[classNum],conf_score,x_1,y_1,x_2,y_2])

               
        with open(bbox_folder+test_img_name[:-3]+'txt', 'w') as file:
            for lineNum in range(len(detecttion_list)):                       
                file.write(detecttion_list[lineNum][0] + ' ')
                file.write(str(detecttion_list[lineNum][1]) + ' ')
                file.write(str(detecttion_list[lineNum][2]) + ' ')
                file.write(str(detecttion_list[lineNum][3]) + ' ')  
                file.write(str(detecttion_list[lineNum][4]) + ' ')
                file.write(str(detecttion_list[lineNum][5]))          
                file.write('\n')

        tiff.imwrite(img_folder+test_img_name[:-3]+'tif',mask_save)

    # mask_256 = np.zeros((5,256,256),dtype=np.uint8)
    # for c in range(5):
    #     mask_256[c] = np.uint8(np.array(Image.fromarray(mask_save[c]).resize((256,256))))
    # mask_256[mask_256>0] = 255
    # tiff.imwrite(img_folder_256+test_img_name[:-3]+'tif',mask_256)


    
   
    # break

           