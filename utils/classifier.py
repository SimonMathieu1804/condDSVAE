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
from sklearn.metrics import r2_score, balanced_accuracy_score
import pickle

from utils import *  

def entropy_Hyx(p, eps=1E-16):
    sum_h = (p * torch.log(p + eps)).sum(axis = 1)
    # average over images
    avg_h = torch.mean(sum_h) * (-1)
    return avg_h
    
def entropy_Hy(p_yx, eps=1E-16):
    p_yx = p_yx.detach().cpu().numpy()
    p_y = p_yx.mean(axis=0)
    sum_h = (p_y * np.log(p_y + eps)).sum() * (-1)
    return sum_h
    
def inception_score(p_yx,  eps=1E-16):
    p_yx = p_yx.detach().cpu().numpy()
    # calculate p(y)
    p_y = np.expand_dims(p_yx.mean(axis=0), 0)
    # kl divergence for each image
    kl_d = p_yx * (np.log(p_yx + eps) - np.log(p_y + eps))
    # sum over classes
    sum_kl_d = kl_d.sum(axis=1)
    # average over images
    avg_kl_d = np.mean(sum_kl_d)
    # undo the logs
    is_score = np.exp(avg_kl_d)
    return is_score

  
# Classifier for dSprites
class conv_class(nn.Module):
    def __init__(self, nin, nout):
        super(conv_class, self).__init__()
        self.main = nn.Sequential(
                nn.Conv2d(nin, nout, 4, 2, 1),
                nn.BatchNorm2d(nout), 
                nn.LeakyReLU(0.2, inplace=True),
                )

    def forward(self, input):
        return self.main(input)

class encoder_class(nn.Module):
    def __init__(self, dim, nc=1):
        super(encoder_class, self).__init__()
        self.dim = dim
        nf = 64
        # input is (nc) x 64 x 64
        self.c1 = conv_class(nc, nf)
        # state size. (nf) x 32 x 32
        self.c2 = conv_class(nf, nf * 2)
        # state size. (nf*2) x 16 x 16
        self.c3 = conv_class(nf * 2, nf * 4)
        # state size. (nf*4) x 8 x 8
        self.c4 = conv_class(nf * 4, nf * 8)
        # state size. (nf*8) x 4 x 4
        self.c5 = nn.Sequential(
                nn.Conv2d(nf * 8, dim, 4, 1, 0),
                nn.BatchNorm2d(dim), #BatchNorm
                nn.Tanh()
                )

    def forward(self, input):
        h1 = self.c1(input)
        h2 = self.c2(h1)
        h3 = self.c3(h2)
        h4 = self.c4(h3)
        h5 = self.c5(h4)
        return h5.view(-1, self.dim), [h1, h2, h3, h4]

class Classifier(nn.Module):                                                                                                                                           
    def __init__(self, g_dim=128, channels=3, rnn_size=256, frames=16):
        super(Classifier, self).__init__()
        self.g_dim = g_dim  # frame feature
        self.channels = channels  # frame feature
        self.hidden_dim = rnn_size
        self.frames = frames
        self.encoder = encoder_class(self.g_dim, self.channels)
        self.bilstm = nn.LSTM(self.g_dim, self.hidden_dim, 1, bidirectional=True, batch_first=True)
        self.rnn = nn.RNN(self.hidden_dim * 2, self.hidden_dim, batch_first=True)
        #cls_color, cls_shape, cls_scale, cls_orientation, cls_positionX,cls_positionY
        self.cls_color = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(True),
            nn.Linear(self.hidden_dim, 6)) 
        self.cls_shape = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(True),
            nn.Linear(self.hidden_dim, 3)) 
        self.cls_scale = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(True),
            nn.Linear(self.hidden_dim, 1)) 
        self.cls_orientation = nn.Sequential(
            nn.Linear(self.hidden_dim*2, self.hidden_dim),
            nn.ReLU(True),
            nn.Linear(self.hidden_dim, 1)) 
        self.cls_positionX = nn.Sequential(
            nn.Linear(self.hidden_dim*2, self.hidden_dim),
            nn.ReLU(True),
            nn.Linear(self.hidden_dim, 1)) 
        self.cls_positionY = nn.Sequential(
            nn.Linear(self.hidden_dim*2, self.hidden_dim),
            nn.ReLU(True),
            nn.Linear(self.hidden_dim, 1)) 
            
    def predict(self, x):
        p_color, p_shape, p_scale, p_orientation, p_positionX, p_positionY = self.forward(x)
        listout = [p_color, p_shape, p_scale, p_orientation, p_positionX, p_positionY]
        return listout
    
    def forward(self, x):
        # input is shape [B,Frames,3,64,64]
        x_shape = x.shape
        x = x.view(-1, x_shape[-3], x_shape[-2], x_shape[-1])
        x_embed = self.encoder(x)[0]
        x_embed = x_embed.view(x_shape[0], x_shape[1], -1) #[B,Frames,N]
        lstm_out, _ = self.bilstm(x_embed)
        backward = lstm_out[:, 0, self.hidden_dim:2 * self.hidden_dim]
        frontal = lstm_out[:, self.frames - 1, 0:self.hidden_dim]
        lstm_out_f = torch.cat((frontal, backward), dim=1) # Shape [B,N]
        p_color, p_shape, p_scale = self.cls_color(lstm_out_f), self.cls_shape(lstm_out_f), self.cls_scale(lstm_out_f)
        p_color, p_shape, p_scale = p_color.unsqueeze(1).expand(-1, self.frames, 6).contiguous(), p_shape.unsqueeze(1).expand(-1, self.frames, 3).contiguous(), p_scale.expand(-1, self.frames).contiguous()
        #features, _ = self.rnn(lstm_out)
        p_orientation, p_positionX, p_positionY = self.cls_orientation(lstm_out).squeeze(), self.cls_positionX(lstm_out).squeeze(),self.cls_positionY(lstm_out).squeeze()
        return p_color, p_shape, p_scale, p_orientation, p_positionX, p_positionY

    def test_output(self,listin,trueLabels):
        # outputs accuracy and R2 score (depending if classes or regression), H, hyx and IS and give the string name for table
        p_color, p_shape, p_scale, p_orientation, p_positionX, p_positionY = listin # the proba are in shape [batchall, frames, n] and consider we have all data
        p_color, p_shape, p_scale, p_orientation, p_positionX, p_positionY = p_color.view(-1, 6), p_shape.view(-1, 3), p_scale.view(-1), p_orientation.view(-1), p_positionX.view(-1), p_positionY.view(-1)
        labels = trueLabels.view(-1,6)
        score_scale = r2_score(labels[:,2].detach().cpu().numpy()/5, p_scale.detach().cpu().numpy())
        score_orien = r2_score(labels[:,3].detach().cpu().numpy()/40, p_orientation.detach().cpu().numpy())
        score_posX = r2_score(labels[:,4].detach().cpu().numpy()/31, p_positionX.detach().cpu().numpy())
        score_posY = r2_score(labels[:,5].detach().cpu().numpy()/31, p_positionY.detach().cpu().numpy())
        score_color = balanced_accuracy_score(labels[:,0].detach().cpu().numpy(), torch.max(p_color, 1)[1].detach().cpu().numpy(), adjusted=True)
        score_shape = balanced_accuracy_score(labels[:,1].detach().cpu().numpy(), torch.max(p_shape, 1)[1].detach().cpu().numpy(), adjusted=True)
        hyx_color,hyx_shape = entropy_Hyx(F.softmax(p_color,dim=1)), entropy_Hyx(F.softmax(p_shape,dim=1))
        hy_color, hy_shape,IS_color,IS_shape = entropy_Hy(F.softmax(p_color,dim=1)), entropy_Hy(F.softmax(p_shape,dim=1)), inception_score(F.softmax(p_color,dim=1)), inception_score(F.softmax(p_shape,dim=1))
        scores_labels = ["score_color","score_shape","score_scale","score_orien","score_posX","score_posY","hyx_color","hyx_shape"]
        scores = [score_color,score_shape,score_scale,score_orien,score_posX,score_posY,hyx_color,hyx_shape]
        diversity_labels = ["hy_color", "hy_shape","IS_color","IS_shape"]
        diversity = [hy_color, hy_shape,IS_color,IS_shape]
        return scores,scores_labels,diversity,diversity_labels
