import os
from PIL import Image
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
from skimage.draw import random_shapes
import skimage
import random
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data import random_split
from skimage.metrics import structural_similarity as ssim
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import sklearn
import pickle


# Make function that gives random label sequence 
def random_label(numframes,conditional):
        # static factors : color, shape, scale
        # dynamic factors : orientation, position X, position Y
        # 1) Simple case : random speed and all ind. factors, orientation fixed
        # 2) Causal case : orientation random, scale change speed and bouncing
        labels = torch.zeros(numframes,6) # 6,3,6,40,32,32 possibilities of factors
        
        # choose static factors
        idx_color = random.randint(0,5)
        idx_shape = random.randint(0,2)
        idx_scale = random.randint(0,5)
        
        # Make rolling vectors for the dynamic factors
        if idx_shape==0:
            roll_orientation = np.linspace(0,9,10,dtype=int)
        elif idx_shape==1:
            roll_orientation = np.linspace(0,19,20,dtype=int)
        else:
            roll_orientation = np.linspace(0,38,39,dtype=int)
        if conditional:
            roll_posx = np.concatenate((np.linspace(0+2*idx_scale,31-2*idx_scale,32-2*2*idx_scale,dtype=int),np.linspace(30-2*idx_scale,1+2*idx_scale,30-2*2*idx_scale,dtype=int)),axis=0)
            roll_posy = np.concatenate((np.linspace(0+2*idx_scale,31-2*idx_scale,32-2*2*idx_scale,dtype=int),np.linspace(30-2*idx_scale,1+2*idx_scale,30-2*2*idx_scale,dtype=int)),axis=0)
        else:
            roll_posx = np.concatenate((np.linspace(0,31,32,dtype=int),np.linspace(30,1,30,dtype=int)),axis=0)
            roll_posy = np.concatenate((np.linspace(0,31,32,dtype=int),np.linspace(30,1,30,dtype=int)),axis=0)
        
        # choose initial value for the dynamic factors
        if conditional:
            roll_orientation = np.roll(roll_orientation,random.randint(0,39))
            roll_posx = np.roll(roll_posx,random.randint(0,31-2*2*idx_scale))
            roll_posy = np.roll(roll_posy,random.randint(0,31-2*2*idx_scale))
        else:
            roll_posx = np.roll(roll_posx,random.randint(0,31))
            roll_posy = np.roll(roll_posy,random.randint(0,31))
        idx_orientation = roll_orientation[0]
        idx_posx = roll_posx[0]
        idx_posy = roll_posy[0]
        
        # store all initinal labels
        idx_tot = idx_shape*32*32*40*6 + idx_scale*32*32*40 + idx_orientation*32*32 + idx_posx*32 + idx_posy
        labels[0] = torch.tensor([idx_color,idx_shape,idx_scale,idx_orientation,idx_posx,idx_posy], dtype=torch.float32)
        
        # initialize the speed for the changes
        if conditional:
            if idx_scale==0 or idx_scale==1:
                speedvec = [5,4,3,2,1,0,-1,-2,-3,-4,-5,-4,-3,-2,-1,0,1,2,3,4]
            elif idx_scale==2 or idx_scale==3:
                speedvec = [4,3,2,1,0,-1,-2,-3,-4,-3,-2,-1,0,1,2,3]
            else:
                speedvec = [3,2,1,0,-1,-2,-3,-2,-1,0,1,2]
            orivariationflag = np.random.rand(1)[0]
            orientation_speed = np.roll(np.array([0]),random.randint(0,8)) if orivariationflag<0.35 else np.roll(np.array([3,2,1,0,-1,-2,-3,-2,-1,0,1,2]),random.randint(0,6))
            posX_speed = np.roll(np.array(speedvec),random.randint(0,8))
            posY_speed = np.roll(np.array(speedvec),random.randint(0,8))
        else:
            orientation_speed = np.roll(np.array([0]),random.randint(0,8))
            posX_speed = np.roll(np.array([4,3,2,1,0,-1,-2,-3,-4,-3,-2,-1,0,1,2,3]),random.randint(0,8))
            posY_speed = np.roll(np.array([4,3,2,1,0,-1,-2,-3,-4,-3,-2,-1,0,1,2,3]),random.randint(0,8))
        
        # loop to make all labels and video
        for i in range(numframes-1):
            roll_orientation = np.roll(roll_orientation,orientation_speed[0])
            orientation_speed = np.roll(orientation_speed,random.choice([0,0,1,-1]))
            idx_orientation = roll_orientation[0]
            roll_posx = np.roll(roll_posx,posX_speed[0])
            posX_speed = np.roll(posX_speed,random.choice([0,0,1,-1]))
            idx_posx = roll_posx[0]
            roll_posy = np.roll(roll_posy,posY_speed[0])
            posY_speed =np.roll(posY_speed,random.choice([0,0,1,-1]))
            idx_posy = roll_posy[0]
            
            idx_tot = idx_shape*32*32*40*6 + idx_scale*32*32*40 + idx_orientation*32*32 + idx_posx*32 + idx_posy
            labels[i+1] = torch.tensor([idx_color,idx_shape,idx_scale,idx_orientation,idx_posx,idx_posy], dtype=torch.float32)
            
        return labels

# Make function that give sameStatic and sameCondDynamic
def make_sameCondDynamic(inlabel,conditional):
        # static factors : color, shape, scale
        # dynamic factors : orientation, position X, position Y
        # 1) Simple case : random speed and all ind. factors, orientation fixed
        # 2) Causal case : orientation random, scale change speed and bouncing
        
        outlabel = torch.clone(inlabel) # inlabel [numframes,6]
        
        if conditional:
            idx_color = random.randint(0,5)
            outlabel[:,0] = torch.ones(frames, dtype=torch.float32)*idx_color
        else:
            idx_color = random.randint(0,5)
            idx_shape = random.randint(0,2)
            idx_scale = random.randint(0,5)
            outlabel[:,0] = torch.ones(frames, dtype=torch.float32)*idx_color
            outlabel[:,1] = torch.ones(frames, dtype=torch.float32)*idx_shape
            outlabel[:,2] = torch.ones(frames, dtype=torch.float32)*idx_scale
            
        return outlabel

def make_sameStaticall(inlabel,conditional):
        # static factors : color, shape, scale
        # dynamic factors : orientation, position X, position Y
        # 1) Simple case : random speed and all ind. factors
        # 2) Causal case : scale change speed and bouncing + different posx/posy change depending on shape + different distribution of orientation change/init with shapes    

        outlabel = torch.clone(inlabel) # inlabel [numframes,6]
        
        # choose static factors
        idx_color = int(inlabel[0,0].item())
        idx_shape = int(inlabel[0,1].item())
        idx_scale = int(inlabel[0,2].item())
        
        # Make rolling vectors for the dynamic factors 
        if idx_shape==0:
            roll_orientation = np.linspace(0,9,10,dtype=int)
        elif idx_shape==1:
            roll_orientation = np.linspace(0,19,20,dtype=int)
        else:
            roll_orientation = np.linspace(0,38,39,dtype=int)
        if conditional:
            roll_posx = np.concatenate((np.linspace(0+2*idx_scale,31-2*idx_scale,32-2*2*idx_scale,dtype=int),np.linspace(30-2*idx_scale,1+2*idx_scale,30-2*2*idx_scale,dtype=int)),axis=0)
            roll_posy = np.concatenate((np.linspace(0+2*idx_scale,31-2*idx_scale,32-2*2*idx_scale,dtype=int),np.linspace(30-2*idx_scale,1+2*idx_scale,30-2*2*idx_scale,dtype=int)),axis=0)
        else:
            roll_posx = np.concatenate((np.linspace(0,31,32,dtype=int),np.linspace(30,1,30,dtype=int)),axis=0)
            roll_posy = np.concatenate((np.linspace(0,31,32,dtype=int),np.linspace(30,1,30,dtype=int)),axis=0)
        
        # choose initial value for the dynamic factors
        if conditional:
            roll_orientation = np.roll(roll_orientation,random.randint(0,39))
            roll_posx = np.roll(roll_posx,random.randint(0,31-2*2*idx_scale))
            roll_posy = np.roll(roll_posy,random.randint(0,31-2*2*idx_scale))
        else:
            roll_posx = np.roll(roll_posx,random.randint(0,31))
            roll_posy = np.roll(roll_posy,random.randint(0,31))
        idx_orientation = roll_orientation[0]
        idx_posx = roll_posx[0]
        idx_posy = roll_posy[0]
        
        # store all initinal outlabel
        idx_tot = idx_shape*32*32*40*6 + idx_scale*32*32*40 + idx_orientation*32*32 + idx_posx*32 + idx_posy
        outlabel[0] = torch.tensor([idx_color,idx_shape,idx_scale,idx_orientation,idx_posx,idx_posy], dtype=torch.float32)
        
        # initialize the speed for the changes
        if conditional:
            if idx_scale==0 or idx_scale==1:
                speedvec = [5,4,3,2,1,0,-1,-2,-3,-4,-5,-4,-3,-2,-1,0,1,2,3,4]
            elif idx_scale==2 or idx_scale==3:
                speedvec = [4,3,2,1,0,-1,-2,-3,-4,-3,-2,-1,0,1,2,3]
            else:
                speedvec = [3,2,1,0,-1,-2,-3,-2,-1,0,1,2]
            orivariationflag = np.random.rand(1)[0]
            orientation_speed = np.roll(np.array([0]),random.randint(0,8)) if orivariationflag<0.35 else np.roll(np.array([3,2,1,0,-1,-2,-3,-2,-1,0,1,2]),random.randint(0,6))
            posX_speed = np.roll(np.array(speedvec),random.randint(0,8))
            posY_speed = np.roll(np.array(speedvec),random.randint(0,8))
        else:
            orientation_speed = np.roll(np.array([0]),random.randint(0,8))
            posX_speed = np.roll(np.array([4,3,2,1,0,-1,-2,-3,-4,-3,-2,-1,0,1,2,3]),random.randint(0,8))
            posY_speed = np.roll(np.array([4,3,2,1,0,-1,-2,-3,-4,-3,-2,-1,0,1,2,3]),random.randint(0,8))
        
        # loop to make all outlabel and video
        for i in range(frames-1):
            roll_orientation = np.roll(roll_orientation,orientation_speed[0])
            orientation_speed = np.roll(orientation_speed,random.choice([0,0,1,-1]))
            idx_orientation = roll_orientation[0]
            roll_posx = np.roll(roll_posx,posX_speed[0])
            posX_speed = np.roll(posX_speed,random.choice([0,0,1,-1]))
            idx_posx = roll_posx[0]
            roll_posy = np.roll(roll_posy,posY_speed[0])
            posY_speed =np.roll(posY_speed,random.choice([0,0,1,-1]))
            idx_posy = roll_posy[0]
            
            idx_tot = idx_shape*32*32*40*6 + idx_scale*32*32*40 + idx_orientation*32*32 + idx_posx*32 + idx_posy
            outlabel[i+1] = torch.tensor([idx_color,idx_shape,idx_scale,idx_orientation,idx_posx,idx_posy], dtype=torch.float32)
            
        return outlabel


# Give parameters for data
trainset_size = 100000       
testset_size = 20000   
frames = 16  

# Make training data
train_simple = []
train_cond = []
for j in range(trainset_size):
    train_simple.append(random_label(frames,False))
    train_cond.append(random_label(frames,True))
train_simple = torch.stack(train_simple)
torch.save(train_simple,"simple/train")
train_cond = torch.stack(train_cond)
torch.save(train_cond,"cond/train")

# Make testing data : 
# 1) testset_size random labels
# 2) testset_size sameCondDynamic
# 3) testset_size sameStaticAll
test_simple = []
sameCondDynamic_simple = []
sameStaticall_simple = []
test_cond = []
sameCondDynamic_cond = []
sameStaticall_cond = []
for j in range(testset_size):
    test_simple.append(random_label(frames,False))
    test_cond.append(random_label(frames,True))
    sameCondDynamic_simple.append(make_sameCondDynamic(test_simple[j],False))
    sameCondDynamic_cond.append(make_sameCondDynamic(test_cond[j],True))
    sameStaticall_simple.append(make_sameStaticall(test_simple[j],False))
    sameStaticall_cond.append(make_sameStaticall(test_cond[j],True))
test_simple =  torch.stack(test_simple)
sameCondDynamic_simple =  torch.stack(sameCondDynamic_simple)
sameStaticall_simple =  torch.stack(sameStaticall_simple)
torch.save(test_simple,"simple/test")
torch.save(sameCondDynamic_simple,"simple/sameCondDynamic")
torch.save(sameStaticall_simple,"simple/sameStaticall")
test_cond =  torch.stack(test_cond)
sameCondDynamic_cond =  torch.stack(sameCondDynamic_cond)
sameStaticall_cond =  torch.stack(sameStaticall_cond)
torch.save(test_cond,"cond/test")
torch.save(sameCondDynamic_cond,"cond/sameCondDynamic")
torch.save(sameStaticall_cond,"cond/sameStaticall")