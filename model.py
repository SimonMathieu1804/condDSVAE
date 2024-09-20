# baseblocks 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision
from torch.autograd import Function
import numpy as np 
import random
import math

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
device = get_device()
print('device available : ',device)

class conv(nn.Module):
    def __init__(self, nin, nout):
        super(conv, self).__init__()
        self.main = nn.Sequential(
                nn.Conv2d(nin, nout, 4, 2, 1),
                nn.GroupNorm(8,nout),
                nn.SiLU(),
                )

    def forward(self, input):
        return self.main(input)


class encoder(nn.Module):
    def __init__(self, dim=256, nc=3):
        super(encoder, self).__init__()
        self.dim = dim
        nf = 64
        # input is (nc) x 64 x 64
        self.c1 = conv(nc, nf)
        # state size. (nf) x 32 x 32
        self.c2 = conv(nf, nf * 2)
        # state size. (nf*2) x 16 x 16
        self.c3 = conv(nf * 2, nf * 4)
        # state size. (nf*4) x 8 x 8
        self.c4 = conv(nf * 4, nf * 8)
        # state size. (nf*8) x 4 x 4
        self.c5 = nn.Sequential(
                nn.Conv2d(nf * 8, dim, 4, 1, 0),
                nn.GroupNorm(8,dim),
                nn.SiLU()
                )

    def forward(self, input):
        h1 = self.c1(input)
        h2 = self.c2(h1)
        h3 = self.c3(h2)
        h4 = self.c4(h3)
        h5 = self.c5(h4)
        return h5.view(-1, self.dim)


class upconv(nn.Module):
    def __init__(self, nin, nout):
        super(upconv, self).__init__()
        self.main = nn.Sequential(
                nn.ConvTranspose2d(nin, nout, 4, 2, 1),
                nn.GroupNorm(8,nout),
                nn.SiLU(),
                )

    def forward(self, input):
        return self.main(input)


class decoder(nn.Module):
    def __init__(self, dim=288, nc=3):
        super(decoder, self).__init__()
        self.dim = dim
        nf = 64
        self.upc1 = nn.Sequential(
                # input is Z, going into a convolution
                nn.ConvTranspose2d(dim, nf * 8, 4, 1, 0),
                nn.GroupNorm(8,nf * 8),
                nn.SiLU()
                )
        # state size. (nf*8) x 4 x 4
        self.upc2 = upconv(nf * 8, nf * 4)
        # state size. (nf*4) x 8 x 8
        self.upc3 = upconv(nf * 4, nf * 2)
        # state size. (nf*2) x 16 x 16
        self.upc4 = upconv(nf * 2, nf)
        # state size. (nf) x 32 x 32
        self.upc5 = nn.Sequential(
                nn.ConvTranspose2d(nf, nc, 4, 2, 1),
                nn.Sigmoid()#nn.Tanh()
                # state size. (nc) x 64 x 64
                )
        #self.merger = nn.Linear(dim,dim)

    def forward(self, input):
        #test = self.merger(input)
        d1 = self.upc1(input.view(-1, self.dim, 1, 1))
        d2 = self.upc2(d1)
        d3 = self.upc3(d2)
        d4 = self.upc4(d3)
        output = self.upc5(d4)
        output = output.view(input.shape[0], input.shape[1], output.shape[1], output.shape[2], output.shape[3])

        return output


class RealNVP(nn.Module):
    def __init__(self, nets, nett, masks, prior):
        super(RealNVP, self).__init__()
        
        self.prior = prior
        self.mask = nn.Parameter(masks, requires_grad=False)
        self.t = torch.nn.ModuleList([nett() for _ in range(len(masks))])
        self.s = torch.nn.ModuleList([nets() for _ in range(len(masks))])
        
    def g(self, z):
        log_det_J, x = z.new_zeros(z.shape[0]).to(device), z
        for i in range(len(self.t)):
            x_ = x*self.mask[i]
            s = self.s[i](x_)*(1 - self.mask[i])
            t = self.t[i](x_)*(1 - self.mask[i])
            x = x_ + (1 - self.mask[i]) * (x * torch.exp(s) + t)
            #log_det_J += s.sum(dim=1)
        return x, log_det_J

    def f(self, x):
        log_det_J, z = x.new_zeros(x.shape[0]).to(device), x
        for i in reversed(range(len(self.t))):
            z_ = self.mask[i] * z
            s = self.s[i](z_) * (1-self.mask[i])
            t = self.t[i](z_) * (1-self.mask[i])
            z = (1 - self.mask[i]) * (z - t) * torch.exp(-s) + z_
            log_det_J -= s.sum(dim=1)
        return z, log_det_J
    
    def log_prob(self,x):
        z, logp = self.f(x)
        return self.prior.log_prob(z) + logp
        
    def sample(self, batchSize): 
        z = self.prior.sample((batchSize, 1)).to(device)
        x, logdJ = self.g(z)
        return x
        
    def interpolate(self, content1, content2, alpha=0.5):
        z1, _ = self.f(content1)
        z2, _ = self.f(content2)
        percentage = (1-cos(alpha*np.pi))/2;
        z_interp = z1*(1-percentage)+z2*percentage
        content_interp, _ = self.g(z_interp) 
        return content_interp
 
        
class static_encoder(nn.Module):
    def __init__(self, frames=16, latent_channel=256, p_dim=256, NF_dim=512):
        super(static_encoder, self).__init__()
        self.frames = frames
        self.latent_channel = latent_channel
        self.p_dim = p_dim
        self.NF_dim = NF_dim
        
        self.linearnet = nn.Sequential(
                nn.Linear(self.latent_channel, self.latent_channel),
                nn.LayerNorm(self.latent_channel),
                nn.GELU(),
                nn.Linear(self.latent_channel, self.latent_channel),
                nn.LayerNorm(self.latent_channel),
            )
        self.meanNet = nn.Sequential(nn.SiLU(), nn.Linear(self.latent_channel, self.p_dim))
        self.varNet = nn.Sequential(nn.SiLU(), nn.Linear(self.latent_channel, self.p_dim))
        
        # prior NF
        nets = lambda: nn.Sequential(nn.Linear(self.p_dim, self.NF_dim), nn.LeakyReLU(), nn.Linear(self.NF_dim, self.NF_dim), nn.LeakyReLU(), nn.Linear(self.NF_dim, self.p_dim), nn.Tanh())
        nett = lambda: nn.Sequential(nn.Linear(self.p_dim, self.NF_dim), nn.LeakyReLU(), nn.Linear(self.NF_dim, self.NF_dim), nn.LeakyReLU(), nn.Linear(self.NF_dim, self.p_dim))
        a = np.zeros(self.p_dim)
        a[1::2] = 1
        b = np.zeros(self.p_dim)
        b[::2] = 1
        masks = torch.from_numpy(np.array([a,b] * 4).astype(np.float32)).to(device)
        priordist = torch.distributions.MultivariateNormal(torch.zeros(self.p_dim).to(device), torch.eye(self.p_dim).to(device))
        self.NFprior = RealNVP(nets, nett, masks, priordist)
    
    def aggregate_posterior(self, mean, logvar):
        var = torch.exp(logvar)
        y = torch.ones_like(var)*1e-6
        var = torch.where(var != float(0), var, y)
        inv_var = torch.pow(var,-1)
        inv_var_tot = torch.sum(inv_var,dim=1)
        var_tot = torch.pow(inv_var_tot,-1)
        inter1 = mean*inv_var
        inter2 = torch.sum(inter1,dim=1)
        mu_tot = inter2*var_tot
        y2 = torch.ones_like(var_tot)*1e-6
        var_tot = torch.where(var_tot != float(0), var_tot, y2)
        logvar_tot = torch.log(var_tot)
        return mu_tot, logvar_tot
          
    def reparameterize(self, mean, logvar, random_sampling=True):
        if random_sampling is True:
            eps = torch.randn_like(logvar)
            std = torch.exp(0.5*logvar)
            z = mean + eps*std
            return z
        else:
            return mean
              
    def forward(self, x): # input of shape [Batch, frames, latent_channel]
        x = x.view(-1, self.latent_channel) # change to [Batch*frames, latent_channel]
        x = self.linearnet(x)
        mean = self.meanNet(x)
        logvar = self.varNet(x)
        mean = mean.view(-1, self.frames ,self.p_dim)
        logvar = logvar.view(-1, self.frames ,self.p_dim)
        p_all = self.reparameterize(mean, logvar, self.training)
        
        mean_tot, logvar_tot = self.aggregate_posterior(mean, logvar)
        p = self.reparameterize(mean_tot, logvar_tot, True)
        loglikprior = self.NFprior.log_prob(p)
        return p, mean_tot, logvar_tot, loglikprior, mean, logvar, p_all

#"""
class conditionalNF(nn.Module):
    def __init__(self, frames=16, latent_channel=256, p_dim=256, lambda_dim=32):
        super(conditionalNF, self).__init__()
        self.frames = frames
        self.latent_channel = latent_channel
        self.p_dim = p_dim
        self.z_dim = lambda_dim
        self.NF_dim = self.z_dim*2
        
        self.nets = lambda: nn.Sequential(nn.Linear(self.z_dim+self.p_dim, self.NF_dim), nn.LeakyReLU(), nn.Linear(self.NF_dim, self.NF_dim), nn.LeakyReLU(), nn.Linear(self.NF_dim, self.z_dim), nn.Tanh())
        self.nett = lambda: nn.Sequential(nn.Linear(self.z_dim+self.p_dim, self.NF_dim), nn.LeakyReLU(), nn.Linear(self.NF_dim, self.NF_dim), nn.LeakyReLU(), nn.Linear(self.NF_dim, self.z_dim))
        a = np.zeros(self.z_dim)
        a[1::2] = 1
        b = np.zeros(self.z_dim)
        b[::2] = 1
        c = np.zeros(self.z_dim)
        self.masks = torch.from_numpy(np.array([c]+[a,b]*4).astype(np.float32)).to(device)
        self.mask = nn.Parameter(self.masks, requires_grad=False)
        self.layertype = ['c']+['a','b']*4
        self.t = torch.nn.ModuleList([self.nett() for _ in range(len(self.masks))])
        self.s = torch.nn.ModuleList([self.nets() for _ in range(len(self.masks))])
        
        self.hidden_dim = self.z_dim*2
        self.z_prior_lstm_ly1 = nn.LSTMCell(self.z_dim, self.hidden_dim)
        self.z_prior_lstm_ly2 = nn.LSTMCell(self.hidden_dim, self.hidden_dim)
        self.z_prior_mean = nn.Linear(self.hidden_dim, self.z_dim)
        self.z_prior_logvar = nn.Linear(self.hidden_dim, self.z_dim)
        
    def g(self, z, pattern):
        z_out = None
        batch_size = z.shape[0]
        z_t = torch.zeros(batch_size, self.z_dim).cuda()
        for j in range(self.frames):
            x = z[:,j,:]
            for i in range(len(self.t)):
                x_ = x*self.mask[i]
                if self.layertype[i]=='c':
                    inputlayer = torch.cat((z_t,pattern),dim=-1)
                else:
                    inputlayer = torch.cat((x_,pattern),dim=-1)
                s = self.s[i](inputlayer)*(1 - self.mask[i])
                t = self.t[i](inputlayer)*(1 - self.mask[i])
                x = x_ + (1 - self.mask[i]) * (x * torch.exp(s) + t)
                if self.layertype[i]=='c':
                    z_t = x 
            if z_out is None:
                z_out = x.unsqueeze(1)
            else:
                z_out = torch.cat((z_out, x.unsqueeze(1)), dim=1)   
        return z_out
    
    def f(self, nf_in):
        x = nf_in[:,:,self.z_dim:]
        p_expand = nf_in[:,:,:self.z_dim]
        x = x.view(-1, self.z_dim)
        p_expand = p_expand.view(-1, self.p_dim)
        log_det_J, z = x.new_zeros(x.shape[0]).to(device), x      
        for i in reversed(range(len(self.t))):
            z_shift = torch.zeros_like(z).view(-1, self.frames,self.z_dim)
            z_shift[:,1:] = z.view(-1, self.frames,self.z_dim)[:,:-1]
            z_shift = z_shift.view(-1, self.z_dim)
            z_ = self.mask[i] * z
            if self.layertype[i]=='c':
                inputlayer = torch.cat((z_shift,p_expand.detach()),dim=-1) 
            else:
                inputlayer = torch.cat((z_,p_expand),dim=-1)
            s = self.s[i](inputlayer) * (1-self.mask[i])
            t = self.t[i](inputlayer) * (1-self.mask[i])
            z = (1 - self.mask[i]) * (z - t) * torch.exp(-s) + z_
            log_det_J -= s.sum(dim=1)
        log_det_J = log_det_J.view(-1,self.frames)    
        z = z.view(-1,self.frames,self.z_dim)
        return z, log_det_J
     
    def sample_z_prior_test(self, z_post, pattern):
        z_out = None
        batch_size = z_post.shape[0]
        z_t = torch.zeros(batch_size, self.z_dim).cuda()
        h_t_ly1 = torch.zeros(batch_size, self.hidden_dim).cuda()
        c_t_ly1 = torch.zeros(batch_size, self.hidden_dim).cuda()
        h_t_ly2 = torch.zeros(batch_size, self.hidden_dim).cuda()
        c_t_ly2 = torch.zeros(batch_size, self.hidden_dim).cuda()

        for i in range(self.frames):
            in_lstm = z_t
            h_t_ly1, c_t_ly1 = self.z_prior_lstm_ly1(in_lstm, (h_t_ly1, c_t_ly1))
            h_t_ly2, c_t_ly2 = self.z_prior_lstm_ly2(h_t_ly1, (h_t_ly2, c_t_ly2))
            z_mean_t = self.z_prior_mean(h_t_ly2)
            z_logvar_t = self.z_prior_logvar(h_t_ly2)
            newpost = z_post[:,i,:]* torch.exp(z_logvar_t) + z_mean_t
            if z_out is None: 
                z_out = newpost.unsqueeze(1)
            else:
                z_out = torch.cat((z_out, newpost.unsqueeze(1)), dim=1)
            z_t = newpost
        return z_out

    def sample_z_prior_train(self, nf_in):
        z_post = nf_in[:,:,self.z_dim:]
        pattern_expand = nf_in[:,:,:self.z_dim]
        z_out = None
        logp = None
        batch_size = z_post.shape[0]
        z_t = torch.zeros(batch_size, self.z_dim).cuda()
        h_t_ly1 = torch.zeros(batch_size, self.hidden_dim).cuda()
        c_t_ly1 = torch.zeros(batch_size, self.hidden_dim).cuda()
        h_t_ly2 = torch.zeros(batch_size, self.hidden_dim).cuda()
        c_t_ly2 = torch.zeros(batch_size, self.hidden_dim).cuda()

        for i in range(self.frames):
            pattern = pattern_expand[:,i]
            in_lstm = z_t
            h_t_ly1, c_t_ly1 = self.z_prior_lstm_ly1(in_lstm, (h_t_ly1, c_t_ly1))
            h_t_ly2, c_t_ly2 = self.z_prior_lstm_ly2(h_t_ly1, (h_t_ly2, c_t_ly2))
            z_mean_t = self.z_prior_mean(h_t_ly2)
            z_logvar_t = self.z_prior_logvar(h_t_ly2)
            newpost = (z_post[:,i,:] - z_mean_t) * torch.exp(-z_logvar_t)
            if z_out is None: 
                z_out = newpost.unsqueeze(1)
                logp = -z_logvar_t.sum(dim=1).unsqueeze(1)
            else: 
                z_out = torch.cat((z_out, newpost.unsqueeze(1)), dim=1)
                logp = torch.cat((logp, -z_logvar_t.sum(dim=1).unsqueeze(1)), dim=1)
            z_t = z_post[:,i,:]
        return z_out, logp
    
    def log_prob(self,nf_input):
        p_expand = nf_input[:,:,:self.z_dim]
        z, logp = self.f(nf_input)
        z2 = torch.cat((p_expand,z),dim=-1)
        samp, logp2 = self.sample_z_prior_train(z2.detach())
        prior = torch.distributions.Normal(torch.tensor(0.,device=device), torch.tensor(1.,device=device))
        loglik = (torch.sum(prior.log_prob(samp))+torch.sum(logp2))*0.0025 + (torch.sum(prior.log_prob(z)) + torch.sum(logp))
        return loglik, z
       
    def sample(self, pattern): 
        z = torch.randn(len(pattern),self.frames,self.z_dim, device=device)        
        z = self.sample_z_prior_test(z, pattern)
        x = self.g(z, pattern)
        return x
        
    def swapping(self, pattern, motion):  
        x = self.g(motion, pattern)
        return x  
    
    def forward(self, nf_input):
      log_pz, lambda_t = self.log_prob(nf_input) 
      kl = - log_pz
      loss_q = kl/nf_input.shape[0]
      return loss_q, lambda_t
#"""

"""
class conditionalNF(nn.Module):
    def __init__(self, frames=16, latent_channel=256, p_dim=256, lambda_dim=32):
        super(conditionalNF, self).__init__()
        self.frames = frames
        self.latent_channel = latent_channel
        self.p_dim = p_dim
        self.z_dim = lambda_dim
        self.NF_dim = self.z_dim*2
        
        self.nets = lambda: nn.Sequential(nn.Linear(self.z_dim+self.p_dim, self.NF_dim), nn.LeakyReLU(), nn.Linear(self.NF_dim, self.NF_dim), nn.LeakyReLU(), nn.Linear(self.NF_dim, self.z_dim), nn.Tanh())
        self.nett = lambda: nn.Sequential(nn.Linear(self.z_dim+self.p_dim, self.NF_dim), nn.LeakyReLU(), nn.Linear(self.NF_dim, self.NF_dim), nn.LeakyReLU(), nn.Linear(self.NF_dim, self.z_dim))
        a = np.zeros(self.z_dim)
        a[1::2] = 1
        b = np.zeros(self.z_dim)
        b[::2] = 1
        self.masks = torch.from_numpy(np.array([a,b] * 4).astype(np.float32)).to(device)
        self.mask = nn.Parameter(self.masks, requires_grad=False)
        self.t = torch.nn.ModuleList([self.nett() for _ in range(len(self.masks))])
        self.s = torch.nn.ModuleList([self.nets() for _ in range(len(self.masks))])
        
        self.hidden_dim = self.z_dim*2
        self.z_prior_lstm_ly1 = nn.LSTMCell(self.z_dim+self.p_dim, self.hidden_dim)
        self.z_prior_lstm_ly2 = nn.LSTMCell(self.hidden_dim, self.hidden_dim)
        self.z_prior_mean = nn.Linear(self.hidden_dim, self.z_dim)
        self.z_prior_logvar = nn.Linear(self.hidden_dim, self.z_dim)
        
    def g(self, z, pattern):
        log_det_J, x = z.new_zeros(z.shape[0]).to(device), z
        p_expand = pattern.unsqueeze(1).expand(-1, self.frames, self.p_dim).contiguous().view(-1,self.p_dim)
        for i in range(len(self.t)):
            x_ = x*self.mask[i]
            inputlayer = torch.cat((x_,p_expand),dim=-1)
            s = self.s[i](inputlayer)*(1 - self.mask[i])
            t = self.t[i](inputlayer)*(1 - self.mask[i])
            x = x_ + (1 - self.mask[i]) * (x * torch.exp(s) + t)
            log_det_J += s.sum(dim=1)
        return x
    
    def f(self, nf_in):
        #p_expand = pattern.unsqueeze(1).expand(-1, self.frames, self.p_dim).contiguous().view(-1,self.p_dim)
        x = nf_in[:,self.z_dim:]
        p_expand = nf_in[:,:self.z_dim]
        log_det_J, z = x.new_zeros(x.shape[0]).to(device), x        
        for i in reversed(range(len(self.t))):
            z_ = self.mask[i] * z
            inputlayer = torch.cat((z_,p_expand),dim=-1)
            s = self.s[i](inputlayer) * (1-self.mask[i])
            t = self.t[i](inputlayer) * (1-self.mask[i])
            z = (1 - self.mask[i]) * (z - t) * torch.exp(-s) + z_
            log_det_J -= s.sum(dim=1)
        return z, log_det_J
     
    def sample_z_prior_test(self, z_post, pattern):
        z_out = None
        logp = None
        batch_size = z_post.shape[0]
        z_t = torch.zeros(batch_size, self.z_dim).cuda()

        for i in range(self.frames):
            # two layer LSTM and two one-layer FC
            in_lstm = torch.cat((z_t,pattern),dim=-1) #z_t
            h_t_ly1 = torch.zeros(batch_size, self.hidden_dim).cuda()
            c_t_ly1 = torch.zeros(batch_size, self.hidden_dim).cuda()
            h_t_ly2 = torch.zeros(batch_size, self.hidden_dim).cuda()
            c_t_ly2 = torch.zeros(batch_size, self.hidden_dim).cuda()
            h_t_ly1, c_t_ly1 = self.z_prior_lstm_ly1(in_lstm, (h_t_ly1, c_t_ly1))
            h_t_ly2, c_t_ly2 = self.z_prior_lstm_ly2(h_t_ly1, (h_t_ly2, c_t_ly2))
            z_mean_t = self.z_prior_mean(h_t_ly2)
            z_logvar_t = self.z_prior_logvar(h_t_ly2)
            newpost = z_post[:,i,:]* torch.exp(z_logvar_t) + z_mean_t
            if z_out is None: # If z_out is none it means z_t is z_1, hence store it in the format [batch_size, 1, z_dim
                z_out = newpost.unsqueeze(1)
                logp = z_logvar_t.sum(dim=1).unsqueeze(1)
            else: # If z_out is not none, z_t is not the initial z and hence append it to the previous z_ts collected in z_out
                z_out = torch.cat((z_out, newpost.unsqueeze(1)), dim=1)
                logp = torch.cat((logp, z_logvar_t.sum(dim=1).unsqueeze(1)), dim=1)
            z_t = newpost
        return z_out

    def sample_z_prior_train(self, nf_in):
        z_post = nf_in[:,:,self.z_dim:]
        pattern_expand = nf_in[:,:,:self.z_dim]
        z_out = None
        logp = None
        batch_size = z_post.shape[0]
        z_t = torch.zeros(batch_size, self.z_dim).cuda()

        for i in range(self.frames):
            # two layer LSTM and two one-layer FC
            pattern = pattern_expand[:,i]
            in_lstm = torch.cat((z_t,pattern.detach()),dim=-1) #z_t
            h_t_ly1 = torch.zeros(batch_size, self.hidden_dim).cuda()
            c_t_ly1 = torch.zeros(batch_size, self.hidden_dim).cuda()
            h_t_ly2 = torch.zeros(batch_size, self.hidden_dim).cuda()
            c_t_ly2 = torch.zeros(batch_size, self.hidden_dim).cuda()
            h_t_ly1, c_t_ly1 = self.z_prior_lstm_ly1(in_lstm, (h_t_ly1, c_t_ly1))
            h_t_ly2, c_t_ly2 = self.z_prior_lstm_ly2(h_t_ly1, (h_t_ly2, c_t_ly2))
            z_mean_t = self.z_prior_mean(h_t_ly2)
            z_logvar_t = self.z_prior_logvar(h_t_ly2)
            newpost = (z_post[:,i,:] - z_mean_t) * torch.exp(-z_logvar_t)
            if z_out is None: # If z_out is none it means z_t is z_1, hence store it in the format [batch_size, 1, z_dim
                z_out = newpost.unsqueeze(1)
                logp = -z_logvar_t.sum(dim=1).unsqueeze(1)
            else: # If z_out is not none, z_t is not the initial z and hence append it to the previous z_ts collected in z_out
                z_out = torch.cat((z_out, newpost.unsqueeze(1)), dim=1)
                logp = torch.cat((logp, -z_logvar_t.sum(dim=1).unsqueeze(1)), dim=1)
            z_t = z_post[:,i,:]
        return z_out, logp

    def log_prob(self,nf_input):
        p_expand = nf_input[:,:,:self.z_dim]
        zin = nf_input.view(-1, self.z_dim*2)
        z, logp = self.f(zin)
        z = z.view(-1,self.frames,self.z_dim)
        z2 = torch.cat((p_expand,z),dim=-1)
        samp, logp2 = self.sample_z_prior_train(z2)
        prior = torch.distributions.Normal(torch.tensor(0.,device=device), torch.tensor(1.,device=device))
        loglik = torch.sum(prior.log_prob(samp))+torch.sum(logp2) +torch.sum(logp)
        return loglik, samp
       
    def sample(self, pattern): 
        z = torch.randn(len(pattern),self.frames,self.z_dim, device=device)        
        z = self.sample_z_prior_test(z, pattern)
        z = z.view(-1,self.z_dim)
        x = self.g(z, pattern)
        x = x.view(-1,self.frames,self.z_dim)
        return x
        
    def swapping(self, pattern, motion):  
        z = self.sample_z_prior_test(motion, pattern)
        z = z.view(-1,self.z_dim)
        x = self.g(z, pattern)
        x = x.view(-1,self.frames,self.z_dim)
        return x

    def forward(self, nf_input):
      log_pz, lambda_t = self.log_prob(nf_input) 
      kl = - log_pz
      loss_q = kl/nf_input.shape[0]
      return loss_q, lambda_t
"""

class DisentangledVAE(nn.Module):
  def __init__(self, dic):
    super(DisentangledVAE, self).__init__()
    self.in_ch = dic["in_ch"]
    self.latent_channel =dic["latent_channel"]
    self.frames = dic["frames"]
    self.p_dim = dic["latent_channel"]
    self.lambda_dim = dic["latent_channel"]
    self.NF_dim = dic["latent_channel"]*2
    self.in_size = 64
    
    # encoder
    self.encoder = encoder(dim=self.latent_channel,nc=self.in_ch)
    # decoder
    self.decoder = decoder(dim=self.latent_channel, nc=self.in_ch)
    # static code encoder
    self.static_encoder = static_encoder(frames=self.frames, latent_channel=self.latent_channel, p_dim=self.p_dim, NF_dim=self.NF_dim)
    # conditional Normalizing FLow
    self.conditionalNF = conditionalNF(frames=self.frames, latent_channel=self.latent_channel, p_dim=self.p_dim, lambda_dim=self.lambda_dim)
    
    self.meanNet = nn.Linear(self.latent_channel, self.latent_channel)
      
  def swapping(self, content, motion):
    x = self.conditionalNF.swapping(content, motion)
    recon_swap = self.decoder(x)
    return recon_swap
   
  def sample_p(self, content_example, motion):
    content = self.static_encoder.NFprior.sample(len(motion))
    content = torch.squeeze(content)
    if len(motion)==1:
        content = content.unsqueeze(0)
    x = self.conditionalNF.swapping(content, motion)
    recon_swap = self.decoder(x)
    return recon_swap
  
  def sample_l(self,content,motion_example):
    x = self.conditionalNF.sample(content)
    recon_swap = self.decoder(x)
    return recon_swap
   
  def forward(self, x):
    # image encoder-decoder
    x = x.view(-1, self.in_ch, self.in_size, self.in_size)
    x = self.encoder(x)
    mean_x = self.meanNet(x)
    logvar_x = torch.zeros_like(x)
    x = self.static_encoder.reparameterize(mean_x,logvar_x,True)
    x = x.view(-1, self.frames, self.latent_channel)
    recon_x = self.decoder(x)
    # obtain the pattern 
    p, mean_p, logvar_p, loglikprior, mean_all, logvar_all, p_all = self.static_encoder(x.detach())
    # make the conditional NF for aggregated
    p_expand = p.unsqueeze(1).expand(-1, self.frames, self.p_dim).contiguous()
    nf_input = torch.cat((p_expand,x.detach()),dim=-1)
    loss_q_lambda, lambda_t = self.conditionalNF(nf_input)
    # make the conditional NF for shuffle
    p_all_rand = p_all[:,torch.randperm(self.frames),:] 
    nf_input_all = torch.cat((p_all_rand,x.detach()),dim=-1)
    recon_all, _= self.conditionalNF(nf_input_all)
    
    return {"recon_all":recon_all,"recon":recon_x,"p":p,"mean_p":mean_p,"logvar_p":logvar_p,"loglikprior":loglikprior,"lambda_t":lambda_t,"loss_q_lambda":loss_q_lambda}


def gaussian_likelihood(x_hat,x,logscale=torch.Tensor([0.0]).to('cuda:0')):
  scale = torch.exp(logscale)
  mean = x_hat
  dist = torch.distributions.Normal(mean, scale)
  # measure prob of seeing image under p(x|z)
  log_pxz = dist.log_prob(x)
  out = torch.sum(log_pxz)/len(log_pxz)
  return out*2
    
def log_bernoulli_prob(x, p=0.5):
    logprob = x * torch.log(torch.clip(p, 1e-9, 1.0)) + (1 - x) * torch.log(torch.clip(1.0 - p, 1e-9, 1.0))
    out = torch.sum(logprob)/len(logprob)
    return out

def kl_divergence(z, mu, std, loglikprior):      
  q = torch.distributions.Normal(mu, std)
  log_qzx = torch.sum(q.log_prob(z),dim=1)
  log_pz = loglikprior
  # kl
  kl = (log_qzx - log_pz)
  out = torch.sum(kl)/len(kl)
  return out


def loss_fn(original_seq,outnet):
    recon_pt, recon_seq, logvar_p, mean_p, p, loglikprior, loss_q_lambda= outnet["recon_all"],outnet["recon"],outnet["logvar_p"],outnet["mean_p"],outnet["p"],outnet["loglikprior"],outnet["loss_q_lambda"]
    recon_error = -gaussian_likelihood(x_hat=recon_seq, x=original_seq)
    std_p = torch.exp(logvar_p/2)
    kl = kl_divergence(p, mean_p, std_p, loglikprior)

    loss = recon_error + kl*0.2 + recon_pt*20.
    #loss = recon_error + kl*0.5 + recon_pt*20.
    # ablated shuffle test : loss = recon_error + kl*0.2 + loss_q_lambda*20.
    
    return loss, {"recon_pt":recon_pt,"recon_loss":recon_error,"kl":kl,"loss_q_lambda":loss_q_lambda}

