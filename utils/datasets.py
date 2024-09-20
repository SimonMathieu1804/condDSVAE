import os
import cv2
import shutil
import time
import torch
import math
from torch.utils.data import Dataset
import torchvision
import torch.utils.data
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torch.utils.data as data
import random
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data import random_split
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import sklearn
import pickle

from utils import *

# dSprites loader
class dSprites(Dataset):
    # static factors : color, shape, scale
    # dynamic factors : orientation, position X, position Y
    # 1) Simple case : orientation fixed, random speed
    # 2) Causal case : orientation random, scale change speed and bouncing
    def __init__(self,size,conditional,testing, pathdataset, pathimgs):
        self.trainset_size = 100000       
        self.testset_size = 20000
        self.numframes = 16  
        self.conditional = conditional
        self.testing = testing
        self.path = pathdataset
        self.imgs = np.load(pathimgs, allow_pickle=True)['imgs']
        self.color_possib = np.array([(75/256, 159/256, 34/256),(236/256, 92/256, 146/256),(125/256, 131/256, 10/256),(108/256, 219/256, 255/256),(244/256, 113/256, 52/256),(160/256, 111/256, 238/256)])
        
        if self.testing:
            self.length = size if size<self.testset_size else self.testset_size
            self.label_all = torch.load(self.path+'test')[:self.length]
            self.label_sameCondDynamic = torch.load(self.path+'sameCondDynamic')[:self.length]
            self.label_sameStaticall =torch.load(self.path+'sameStaticall')[:self.length] 
        else:
            self.length = size if size<self.trainset_size else self.trainset_size
            self.label_all = torch.load(self.path+'train')[:self.length]

    def __len__(self):
        return self.length
    
    def makesequence(self, inlabel):
        # outputs the sequnce with inlabel (make sequence in parallel)
        video = torch.zeros(self.numframes,3,64,64)
        idx_color = inlabel[:,0].detach().cpu().numpy().astype(int)
        idx_shape = inlabel[:,1].detach().cpu().numpy().astype(int)
        idx_scale = inlabel[:,2].detach().cpu().numpy().astype(int)
        idx_orientation = inlabel[:,3].detach().cpu().numpy().astype(int)
        idx_posx = inlabel[:,4].detach().cpu().numpy().astype(int)
        idx_posy = inlabel[:,5].detach().cpu().numpy().astype(int)
        idx_tot = idx_shape*32*32*40*6 + idx_scale*32*32*40 + idx_orientation*32*32 + idx_posx*32 + idx_posy
        video[:,0] = torch.tensor(self.imgs[idx_tot], dtype=torch.float)*(self.color_possib[idx_color,0][0])
        video[:,1] = torch.tensor(self.imgs[idx_tot], dtype=torch.float)*(self.color_possib[idx_color,1][0])
        video[:,2] = torch.tensor(self.imgs[idx_tot], dtype=torch.float)*(self.color_possib[idx_color,2][0])
        return video
    
    def __getitem__(self, idx):
        # label have shape [Dataset,16,6] with datasetmax=self.length
        if self.testing:
            labelout = self.label_all[idx]
            videoout = self.makesequence(labelout)
            videosameStaticall = self.makesequence(self.label_sameStaticall[idx])
            videosameCondDyn = self.makesequence(self.label_sameCondDynamic[idx])
            return videoout, labelout, videosameCondDyn, videosameStaticall
        else: 
            labelout = self.label_all[idx]
            videoout = self.makesequence(labelout)
            return videoout, labelout