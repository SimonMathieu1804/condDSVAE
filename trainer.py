import os
import shutil
import cv2
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
import wandb
torch.cuda.empty_cache()

from utils.utils import *
from utils.datasets import *
from utils.classifier import *
from model import *
device = get_device()
print('device available : ',device)

# take the arguments for training and testing
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--num_epochs', default=30, type=int, help='number of epochs to train')
parser.add_argument('--frames', default=16, type=int, help='number of frames in sequences')
parser.add_argument('--dataset_size', default=100000, type=int, help='dataset train size')
parser.add_argument('--test_size', default=20000, type=int, help='dataset test size')
parser.add_argument('--conditional', default=True, type=bool, help='if conditional or not dSprites')
parser.add_argument('--batch_size', default=64, type=int, help='training batch size')
parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate')
parser.add_argument('--Gradient_clip', default=True, type=bool, help='wether use gradient clipping or not')
parser.add_argument('--in_ch', default=3, type=int, help='number of channels of the frames')
parser.add_argument('--latent_channel', default=128, type=int, help='size of the latent representation for each frame')
parser.add_argument('--checkpoint_path', default="datasets/classifier_model.pth" , type=str, help='path to the parameters of the classifier')
parser.add_argument('--save_wandb_dir', default="save/wandb/", type=str, help='wandb directory')
parser.add_argument('--save_model_dir', default="save/model/", type=str, help='where to save the model parameters')
parser.add_argument('--pathimgs', default="datasets/dsprites_ndarray.npz", type=str, help='path to the dSprites dataset')
parser.add_argument('--starter_name', default='', type=str, help='name of the run to load already train model')
parser.add_argument('--directory_name', default="condDSVAE" , type=str, help='name of the working directory')
parser.add_argument('--deterministic', default=True, type=bool, help='if seed or not')
parser.add_argument('--model_name', default="condDSVAE" , type=str, help='name of the model')
parser.add_argument('--dataset_name', default="CausaldSprites" , type=str, help='name of the dataset')
opt = parser.parse_args()

num_epochs, frames, dataset_size, test_size, conditional = opt.num_epochs, opt.frames, opt.dataset_size, opt.test_size, opt.conditional
batch_size,learning_rate, Gradient_clip, in_ch, latent_channel = opt.batch_size, opt.learning_rate, opt.Gradient_clip, opt.in_ch, opt.latent_channel
checkpoint_path, save_wandb_dir, save_model_dir, pathimgs = opt.checkpoint_path, opt.save_wandb_dir, opt.save_model_dir, opt.pathimgs
starter_name = None if (opt.starter_name == '') else opt.starter_name
directory_name, deterministic, model_name, dataset_name = opt.directory_name, opt.deterministic, opt.model_name, opt.dataset_name

# Set up parameters
pathdataset = "datasets/cond/" if conditional else "datasets/simple/"
run_name = dataset_name + "_" + model_name
p_dim, lambda_dim = latent_channel, latent_channel
network_input = {"in_ch":in_ch,"latent_channel":latent_channel,"frames":frames}

# Model lightning
class LitModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.network = DisentangledVAE(network_input)
        pytorch_total_params = sum(p.numel() for p in self.network.parameters() if p.requires_grad)
        print('number of parameters : ',pytorch_total_params)
        self.save_hyperparameters()
        
        self.classifier = Classifier()
        checkpoint = torch.load(checkpoint_path)
        self.classifier.load_state_dict(checkpoint['state_dict'])
        self.classifier.eval()
        
        self.test_end = []

    def forward(self, x):
        out = self.network(x)
        return out
    
    def configure_optimizers(self):
        optimizer = optim.Adam(filter(lambda p: p.requires_grad,self.parameters()),lr=learning_rate)
        return [optimizer]
    
    def training_step(self, batch, batch_idx):
        data, label = batch
        out = self.network(data)
        loss, terms = loss_fn(data,out)
        self.log("train_loss/loss", loss, on_step=False,on_epoch=True)
        for key, value in terms.items():
            self.log("train_loss/"+key, value, on_step=False,on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        # Plot the reconstruction error (for test set)
        if batch_idx <=10: 
            images, label = batch
            out = self.network(images)
            loss_val = loss_rec(images,out["recon"])
            self.log("val_loss/loss", loss_val, on_step=False,on_epoch=True)
            if (self.current_epoch+1)%2==0:
                # Plot reconstruction of an image from test set
                idx = random.randint(0,images.shape[0]-1)
                wandb_logger.log_image(key="images/reconstruction", images=[torch.clamp(F.pad(images[idx], (1, 1, 1, 1), "constant", 1) ,min=0, max=1), torch.clamp(F.pad(out["recon"][idx], (1, 1, 1, 1), "constant", 1) ,min=0, max=1)], caption=["original", "reconstructed"])
                # Plot swapping image
                idx2 = random.randint(0, images.shape[0]-1)
                swapped_im = torch.squeeze(self.network.swapping(torch.unsqueeze(out["p"][idx],0),torch.unsqueeze(out["lambda_t"][idx2],0)))
                wandb_logger.log_image(key="images/swapping", images=[torch.clamp(F.pad(images[idx], (1, 1, 1, 1), "constant", 1) ,min=0, max=1), torch.clamp(F.pad(images[idx2], (1, 1, 1, 1), "constant", 1) ,min=0, max=1),torch.clamp(F.pad(swapped_im, (1, 1, 1, 1), "constant", 1) ,min=0, max=1)], caption=["content","motion","swapped"])
                # Plot sample pattern
                sampled_pattern = torch.squeeze(self.network.sample_p(torch.unsqueeze(out["p"][idx],0),torch.unsqueeze(out["lambda_t"][idx2],0)))
                wandb_logger.log_image(key="images/sampling_p", images=[torch.clamp(F.pad(sampled_pattern, (1, 1, 1, 1), "constant", 1) ,min=0, max=1)], caption=["sampled pattern"])
                # Plot sample for motion 
                sampled_motion = torch.squeeze(self.network.sample_l(torch.unsqueeze(out["p"][idx],0),torch.unsqueeze(out["lambda_t"][idx2],0)))
                wandb_logger.log_image(key="images/sampling_lambda", images=[torch.clamp(F.pad(sampled_motion, (1, 1, 1, 1), "constant", 1) ,min=0, max=1)], caption=["sampled motion"]) 
                
    def test_step(self, batch, batch_idx):
        images, labels, sameCondDyn, sameStaticall = batch
        
        out = self.network(images)
        out_sameCondDyn = self.network(sameCondDyn)
        out_sameStaticall = self.network(sameStaticall)
        out_sameAll = self.network(images)

        # 1) reconstruction loss ===================================
        loss_test = loss_rec(images,out["recon"])
        
        # 2) Classifier losses for sampling and swapping ===================================
        # Sampling p to see diversity
        recon_memory = self.network.sample_p(out["p"],out["lambda_t"])
        list_sampp = self.classifier.predict(recon_memory)        
        # Sampling l to see diversity
        recon_memory = self.network.sample_l(out["p"],out["lambda_t"])
        list_sampl = self.classifier.predict(recon_memory)
        # Swapping random to see if keep static factors
        recon_memory = self.network.swapping(out["p"],torch.roll(out["lambda_t"],1,0))
        list_swaprand = self.classifier.predict(recon_memory)
        # Swapping sameCond to see if keep dynamic factors
        recon_memory = self.network.swapping(out_sameStaticall["p"],out_sameCondDyn["lambda_t"])
        list_swapsamecond = self.classifier.predict(recon_memory)
        
        # 3) Plots and gifs for recon,sampling and swapping ================================
        # plot reconstruction
        if batch_idx <=100:
            idx = 0
            wandb_logger.log_image(key="test/reconstruction", images=[torch.clamp(F.pad(images[idx], (1, 1, 1, 1), "constant", 1),min=0, max=1), torch.clamp(F.pad(out["recon"][idx], (1, 1, 1, 1), "constant", 1),min=0, max=1)], caption=["original", "reconstructed"])
            # plot sample static
            sampled_pattern = []
            tt = []
            for i in range(5):
                inter = torch.squeeze(self.network.sample_p(torch.unsqueeze(out["p"][idx],0),torch.unsqueeze(out["lambda_t"][idx],0)))
                sampled_pattern.append(torch.clamp(F.pad(inter, (1, 1, 1, 1), "constant", 1),min=0, max=1))
                tt.append("sampled pattern "+str(i))
            wandb_logger.log_image(key="test/sampling_p", images=sampled_pattern, caption=tt)
            # plot sample motion 
            sampled_pattern = []
            tt = []
            for i in range(5):
                inter = torch.squeeze(self.network.sample_l(torch.unsqueeze(out["p"][idx],0),torch.unsqueeze(out["lambda_t"][idx],0)))
                sampled_pattern.append(torch.clamp(F.pad(inter, (1, 1, 1, 1), "constant", 1),min=0, max=1))
                tt.append("sampled motion "+str(i))
            wandb_logger.log_image(key="test/sampling_lambda", images=sampled_pattern, caption=tt)
            # plot swapping
            idx2 = 1
            swapped_im = torch.squeeze(self.network.swapping(torch.unsqueeze(out["p"][idx],0),torch.unsqueeze(out["lambda_t"][idx2],0)))
            wandb_logger.log_image(key="test/swapping", images=[torch.clamp(F.pad(images[idx], (1, 1, 1, 1), "constant", 1),min=0, max=1), torch.clamp(F.pad(images[idx2], (1, 1, 1, 1), "constant", 1),min=0, max=1),torch.clamp(F.pad(swapped_im, (1, 1, 1, 1), "constant", 1),min=0, max=1)], caption=["content","motion","swapped"])
            # plot sameMotion and sameContent sequences
            wandb_logger.log_image(key="test/augmentedVideo", images=[torch.clamp(F.pad(images[idx], (1, 1, 1, 1), "constant", 1),min=0, max=1), torch.clamp(F.pad(sameCondDyn[idx], (1, 1, 1, 1), "constant", 1),min=0, max=1),torch.clamp(F.pad(sameStaticall[idx], (1, 1, 1, 1), "constant", 1),min=0, max=1)], caption=["original","sameMotion","sameContent"])
            
        outputtest =  {"reconloss":loss_test,
                "list_sampp":list_sampp,
                "list_sampl":list_sampl,
                "list_swaprand":list_swaprand,
                "list_swapsamecond":list_swapsamecond,
                "labels":labels,
                }
        self.test_end.append(outputtest)
        return outputtest 

    def on_test_epoch_end(self):
        
        outputs = self.test_end
        columns = ["model"]
        data = [model_name]
        
        # 1) reconloss
        average_value = sum(d["reconloss"] for d in outputs) / len(outputs)
        columns.append("reconloss")
        data.append(average_value)
        
        # 2) classifier metrics
        # scores,scores_labels,diversity,diversity_labels = self.classifier.test_output(listin,trueLabels) 
        # list_sampp,list_sampl,list_swaprand,list_swapsamecond,labels
        
        # extract labels
        truelabels = torch.cat([d["labels"] for d in outputs],dim=0)
        # exctract list_sampp
        inter = [d["list_sampp"] for d in outputs]
        examplelist = inter[0]
        list_sampp = []
        for i in range(len(examplelist)):
            list_sampp.append(torch.cat([inter[j][i] for j in range(len(inter))],dim=0))
        _,_,diversity,diversity_labels = self.classifier.test_output(list_sampp,truelabels) 
        columns += ["sampp_"+diversity_labels[i] for i in range(len(diversity_labels))]
        data += diversity
        # exctract list_sampl
        inter = [d["list_sampl"] for d in outputs]
        examplelist = inter[0]
        list_sampl = []
        for i in range(len(examplelist)):
            list_sampl.append(torch.cat([inter[j][i] for j in range(len(inter))],dim=0))
        _,_,diversity,diversity_labels = self.classifier.test_output(list_sampl,truelabels) 
        columns += ["sampl_"+diversity_labels[i] for i in range(len(diversity_labels))]
        data += diversity
        # exctract list_swaprand
        inter = [d["list_swaprand"] for d in outputs]
        examplelist = inter[0]
        list_swaprand = []
        for i in range(len(examplelist)):
            list_swaprand.append(torch.cat([inter[j][i] for j in range(len(inter))],dim=0))
        scores,scores_labels,_,_ = self.classifier.test_output(list_swaprand,truelabels) 
        columns += ["swaprand_"+scores_labels[i] for i in range(len(scores_labels))]
        data += scores
        # exctract list_swapsamecond
        inter = [d["list_swapsamecond"] for d in outputs]
        examplelist = inter[0]
        list_swapsamecond = []
        for i in range(len(examplelist)):
            list_swapsamecond.append(torch.cat([inter[j][i] for j in range(len(inter))],dim=0))
        scores,scores_labels,_,_ = self.classifier.test_output(list_swapsamecond,truelabels) 
        columns += ["swapsamecond_"+scores_labels[i] for i in range(len(scores_labels))]
        data += scores
        
        # 3) print table
        wandb_logger.log_table(key="disentanglement metrics "+dataset_name, columns=columns, data=[data])


if __name__ == '__main__':

    # Training part
    if deterministic:
        seed_everything(42, workers=True)
    lit = LitModel()
    wandb_logger = WandbLogger(project=directory_name,name=run_name,group=run_name,save_dir=save_wandb_dir)
    checkpoint_callback = ModelCheckpoint(dirpath=save_model_dir+run_name+"/",filename=run_name)
    trainer = Trainer(logger=wandb_logger, accelerator="auto",gradient_clip_val=0.5, enable_progress_bar=False, profiler='simple', max_epochs = num_epochs, default_root_dir='logs/'+model_name,callbacks=[checkpoint_callback],deterministic=deterministic)
    data = dSprites(size=dataset_size,conditional=True,testing=False,pathdataset=pathdataset,pathimgs=pathimgs)
    loader_train = DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=4)
    loader_val = DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=4)
    if starter_name==None:
        if os.path.exists(save_model_dir+run_name) and os.path.isdir(save_model_dir+run_name):
            shutil.rmtree(save_model_dir+run_name)
        trainer.fit(lit, loader_train, loader_val)
    else:
        trainer.fit(lit, loader_train, loader_val, ckpt_path=save_model_dir+run_name+"/"+starter_name+'.ckpt')

    # Testing part
    print("Starting test") 
    data = dSprites(size=test_size,conditional=True,testing=True,pathdataset=pathdataset,pathimgs=pathimgs)
    loader_test = DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=4)
    trainer.test(model=lit,dataloaders=loader_test)
