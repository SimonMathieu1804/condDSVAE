import os
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

def get_device():
    if torch.cuda.is_available():
        device = 'cuda:0'
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')
    else:
        device = 'cpu'
    return device

def loss_rec(original_seq,recon_seq):
    batch_size = original_seq.size(0)
    mse = F.mse_loss(recon_seq,original_seq,reduction='sum');
    return (mse)/batch_size
    
def save_model(model, optim, epoch, path):
    torch.save({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer': optim.state_dict()}, path)
 
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

def logsumexp(value, dim=None, keepdim=False):
    """Numerically stable implementation of the operation
    value.exp().sum(dim, keepdim).log()
    """
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + torch.log(torch.sum(torch.exp(value0),dim=dim, keepdim=keepdim))
    else:
        raise ValueError('Must specify the dimension.')
 
def log_density(x, mu, logvar, scale_mean, scale_std, pattern=None, NFmodel=None):   
    if NFmodel == None:
        latent_sample, logdJ = torch.clone(x).detach(), torch.zeros(len(x),device=device)
    else:
        latent_sample, logdJ = NFmodel.swapping(pattern, x)
    if x.ndim == 3:
        batchsize, numframes, hidden_dim = x.shape
        latent_sample, mu, logvar = latent_sample.view(batchsize,numframes*hidden_dim), mu.view(batchsize,numframes*hidden_dim), logvar.view(batchsize,numframes*hidden_dim)
    #latent_sample2 = torch.clone((latent_sample - scale_mean)/scale_std).detach()
    #mu2 = torch.clone((mu - scale_mean)/scale_std).detach()
    #normalization = - 0.5 * (math.log(2 * math.pi) + torch.zeros_like(logvar))
    inv_var = torch.exp(-torch.zeros_like(logvar)) 
    log_density = -(((latent_sample - mu)/scale_std)**2 * inv_var)#normalization - 0.5 * ((latent_sample - mu)**2 * inv_var)
    return log_density.mean(dim=1) #+ logdJ #output of size [B]

def matrix_log_density(x, mu, logvar,scale_mean, scale_std, pattern=None, NFmodel=None):
    if x.ndim == 3:
        batchsize, numframes, hidden_dim = x.shape
        x = x.unsqueeze(1).expand(batchsize, batchsize, numframes, hidden_dim).contiguous().view(-1,numframes, hidden_dim)
        mu = mu.unsqueeze(0).expand(batchsize, batchsize, numframes, hidden_dim).contiguous().view(-1,numframes, hidden_dim)
        logvar = logvar.unsqueeze(0).expand(batchsize, batchsize, numframes, hidden_dim).contiguous().view(-1,numframes, hidden_dim)
    else:
        batchsize, hidden_dim = x.shape
        x = x.unsqueeze(1).expand(batchsize, batchsize, hidden_dim).contiguous().view(-1,hidden_dim)
        mu = mu.unsqueeze(0).expand(batchsize, batchsize, hidden_dim).contiguous().view(-1,hidden_dim)
        logvar = logvar.unsqueeze(0).expand(batchsize, batchsize, hidden_dim).contiguous().view(-1,hidden_dim)
    if pattern != None:
        pattern = pattern.unsqueeze(0).expand(batchsize, batchsize, p_dim).contiguous().view(-1,p_dim)
    out = log_density(x,mu,logvar,scale_mean, scale_std,pattern,NFmodel).view(batchsize, batchsize)
    return out #output of size [B,B]
    
def negNotmean_Hz_x(x, mu, logvar, pattern=None, NFmodel=None):
    x = torch.clone(x).detach()
    mu = torch.clone(mu).detach()
    logvar = torch.clone(logvar).detach()
    pattern = pattern if pattern == None else torch.clone(pattern).detach()
    if x.ndim == 3:
        batchsize, numframes, hidden_dim = x.shape
        scale_mean = x.mean(0,keepdim=True)
        scale_std = x.std(0,unbiased=False,keepdim=True)
        scale_mean = scale_mean.view(1,numframes*hidden_dim)
        scale_std = scale_std.view(1,numframes*hidden_dim)
    else:
        batchsize, hidden_dim = x.shape
        scale_mean = x.mean(0,keepdim=True)
        scale_std = x.std(0,unbiased=False,keepdim=True)
    return log_density(x, mu, logvar, scale_mean, scale_std, pattern, NFmodel)

def negNotmean_Hz(x, mu, logvar, x2=None, mu2=None, logvar2=None, pattern=None, NFmodel=None, pattern2=None, NFmodel2=None):
    x = torch.clone(x).detach()
    mu = torch.clone(mu).detach()
    logvar = torch.clone(logvar).detach()
    pattern = pattern if pattern == None else torch.clone(pattern).detach()
    x2 = x2 if x2 ==None else torch.clone(x2).detach()
    mu2 = mu2 if mu2==None else torch.clone(mu2).detach()
    logvar2 = logvar2 if logvar2 ==None else torch.clone(logvar2).detach()
    pattern2 = pattern2 if pattern2 == None else torch.clone(pattern2).detach()
    # if both p and lambda at same time p must be put for x2
    batchsize = len(x)
    if x.ndim == 3:
        batchsize, numframes, hidden_dim = x.shape
        scale_mean = x.mean(0,keepdim=True)
        scale_std = x.std(0,unbiased=False,keepdim=True)
        scale_mean = scale_mean.view(1,numframes*hidden_dim)
        scale_std = scale_std.view(1,numframes*hidden_dim)
    else:
        batchsize, hidden_dim = x.shape
        scale_mean = x.mean(0,keepdim=True)
        scale_std = x.std(0,unbiased=False,keepdim=True)
    mat_log_qz = matrix_log_density(x, mu, logvar,scale_mean, scale_std, pattern, NFmodel)
    if x2 != None:
        if x2.ndim == 3:
            batchsize, numframes, hidden_dim = x2.shape
            scale_mean2 = x2.mean(0,keepdim=True)
            scale_std2 = x2.std(0,unbiased=False,keepdim=True)
            scale_mean2 = scale_mean2.view(1,numframes*hidden_dim)
            scale_std2 = scale_std2.view(1,numframes*hidden_dim)
        else:
            batchsize, hidden_dim = x2.shape
            scale_mean2 = x2.mean(0,keepdim=True)
            scale_std2 = x2.std(0,unbiased=False,keepdim=True)
        mat_log_qz += matrix_log_density(x2, mu2, logvar2,scale_mean2, scale_std2, pattern2, NFmodel2)
        #mat_log_qz = mat_log_qz/2
    log_qz = logsumexp(mat_log_qz, dim=1, keepdim=False) - math.log(batchsize)
    return log_qz

class silhouette():

    @staticmethod
    def score(X, labels, loss=False):
        """Compute the mean Silhouette Coefficient of all samples.
        The Silhouette Coefficient is calculated using the mean intra-cluster
        distance (a) and the mean nearest-cluster distance (b) for each sample.
        The Silhouette Coefficient for a sample is ``(b - a) / max(a, b)``.
        To clarrify, b is the distance between a sample and the nearest cluster
        that b is not a part of.
        This function returns the mean Silhoeutte Coefficient over all samples.
        The best value is 1 and the worst value is -1. Values near 0 indicate
        overlapping clusters. Negative values generally indicate that a sample has
        been assigned to the wrong cluster, as a different cluster is more similar.
	Code developed in NumPy by Alexandre Abraham:
	https://gist.github.com/AlexandreAbraham/5544803  Avatar
        Parameters
        ----------
        X : array [n_samples_a, n_features]
            Feature array.
        labels : array, shape = [n_samples]
                 label values for each sample
        loss : Boolean
                If True, will return negative silhouette score as 
                torch tensor without moving it to the CPU. Can therefore 
                be used to calculate the gradient using autograd.
                If False positive silhouette score as float 
                on CPU will be returned.
        Returns
        -------
        silhouette : float
            Mean Silhouette Coefficient for all samples.
        References
        ----------
        Peter J. Rousseeuw (1987). "Silhouettes: a Graphical Aid to the
            Interpretation and Validation of Cluster Analysis". Computational
            and Applied Mathematics 20: 53-65. doi:10.1016/0377-0427(87)90125-7.
        http://en.wikipedia.org/wiki/Silhouette_(clustering)
        """

        if type(labels) != type(torch.HalfTensor()):
            labels = torch.HalfTensor(labels)
        if not labels.is_cuda:
            labels = labels.cuda()

        if type(X) != type(torch.HalfTensor()):
            X = torch.HalfTensor(X)
        if not X.is_cuda:
            X = X.cuda()

        unique_labels = torch.unique(labels)

        A = silhouette._intra_cluster_distances_block(X, labels, unique_labels)
        B = silhouette._nearest_cluster_distance_block(X, labels, unique_labels)
        sil_samples = (B - A) / torch.maximum(A, B)

        # nan values are for clusters of size 1, and should be 0
        mean_sil_score = torch.mean(torch.nan_to_num(sil_samples))
        if loss:
            return - mean_sil_score
        else:
            return float(mean_sil_score.cpu().numpy())

    @staticmethod
    def _intra_cluster_distances_block(X, labels, unique_labels):
        """Calculate the mean intra-cluster distance.
        Parameters
        ----------
        X : array [n_samples_a, n_features]
            Feature array.
        labels : array, shape = [n_samples]
            label values for each sample
        Returns
        -------
        a : array [n_samples_a]
            Mean intra-cluster distance
        """
        intra_dist = torch.zeros(labels.size(), dtype=torch.float32,
                                 device=torch.device("cuda"))
        values = [silhouette._intra_cluster_distances_block_(
                    X[torch.where(labels == label)[0]])
                    for label in unique_labels]
        for label, values_ in zip(unique_labels, values):
            intra_dist[torch.where(labels == label)[0]] = values_
        return intra_dist

    @staticmethod
    def _intra_cluster_distances_block_(subX):
        distances = torch.cdist(subX, subX)
        return distances.sum(axis=1) / (distances.shape[0] - 1)

    @staticmethod
    def _nearest_cluster_distance_block(X, labels, unique_labels):
        """Calculate the mean nearest-cluster distance for sample i.
        Parameters
        ----------
        X : array [n_samples_a, n_features]
            Feature array.
        labels : array, shape = [n_samples]
            label values for each sample
        X : array [n_samples_a, n_features]
            Feature array.
        Returns
        -------
        b : float
            Mean nearest-cluster distance for sample i
        """
        inter_dist = torch.full(labels.size(), float("Inf"),
                                 dtype=torch.float32,
                                device=torch.device("cuda"))
        # Compute cluster distance between pairs of clusters

        label_combinations = torch.combinations(unique_labels, 2)

        values = [silhouette._nearest_cluster_distance_block_(
                    X[torch.where(labels == label_a)[0]],
                    X[torch.where(labels == label_b)[0]])
                    for label_a, label_b in label_combinations]

        for (label_a, label_b), (values_a, values_b) in \
                zip(label_combinations, values):

                indices_a = torch.where(labels == label_a)[0]
                inter_dist[indices_a] = torch.minimum(values_a, inter_dist[indices_a])
                del indices_a
                indices_b = torch.where(labels == label_b)[0]
                inter_dist[indices_b] = torch.minimum(values_b, inter_dist[indices_b])
                del indices_b
        return inter_dist

    @staticmethod
    def _nearest_cluster_distance_block_(subX_a, subX_b):
        dist = torch.cdist(subX_a, subX_b)
        dist_a = dist.mean(axis=1)
        dist_b = dist.mean(axis=0)
        return dist_a, dist_b
 
def CosineInterpolate(x1,x2):
    out = []
    mu = torch.linspace(0,1,20)
    for i in mu:
        mu2 =  (1-np.cos(i*np.pi))/2
        out.append(x1*(1-mu2)+x2*mu2)
    return torch.stack(out)