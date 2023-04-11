#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch
from data_util import toDeviceDataLoader, load_cifar, to_device, load_mnist
from model_util import VGG
import torchvision
from util import dsa
from tqdm.notebook import tqdm
import torch.nn.functional as F
from matplotlib import pyplot as plt
import statsmodels.api as sm
import scipy
import numpy as np
import random

#device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
device = torch.device('cpu')

torch.manual_seed(123)
random.seed(123)
np.random.seed(123)


# In[2]:


class MNet(torch.nn.Module):
    def __init__(self):
        super(MNet, self).__init__()
        self.l1 = torch.nn.Linear(28*28, 100)
        self.l2 = torch.nn.Linear(100, 50)
        self.l3 = torch.nn.Linear(50, 10)
    
    def forward(self, x):
        out = torch.relu(self.l1(x))
        out = torch.relu(self.l2(out))
        return self.l3(out)

mdl = to_device(MNet(), device)
mdl.load_state_dict(torch.load('./models/torch_mnist_net.pth'))
mdl = mdl.eval()

# mdl = to_device(MNet(), device)
# crit = torch.nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(mdl.parameters(), lr=0.01) 
# for epoch in range(10):
#     pbar = tqdm(train_loader)
#     for batch_id, (images, labels) in enumerate(pbar):  
#         outputs = mdl(images.view(-1, 28*28))
#         loss = F.cross_entropy(outputs, labels)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step() 
#         if batch_id == len(train_loader) - 1:
#             pbar.set_postfix({"Test Accuracy":dsa(test_loader, mdl)})
# torch.save(mdl.state_dict(), "models/torch_mnist_net.pth")

train_loader, test_loader = load_mnist(64, 10, device)
# mdl = to_device(VGG('VGG16'), device)
# mdl.load_state_dict(torch.load('./models/torch_cifar_vgg.pth'))
# mdl = mdl.eval()

# dataset_root = '/share/datasets/cifar10'
# cifar10_train, cifar10_val, cifar10_test = load_cifar(dataset_root)
# train_loader, val_loader, test_loader = toDeviceDataLoader(cifar10_train, cifar10_val, cifar10_test, device = device)


# In[3]:


def tnmean_(alphas):
    less = - np.sqrt(2/np.pi) * torch.exp(-(alphas ** 2)/2) / (torch.erf(alphas/np.sqrt(2)) - 1)
    more = np.sqrt(2/np.pi) / torch.special.erfcx(alphas/np.sqrt(2))
    res = torch.where(alphas <= 0, less, more)
    return torch.max(alphas, res)

def tnmean(means, stds, mins):
    alphas = (mins - means)/stds
    return means + tnmean_(alphas) * stds

def tnmom2(alphas):
    return 1 + np.sqrt(2/np.pi) * alphas / torch.special.erfcx(alphas/np.sqrt(2))

def tnvar_(alphas):
    m1 = tnmean_(alphas)
    m2 = torch.sqrt(tnmom2(alphas))
    return (m2 - m1) * (m2 + m1)

def tnvar(means, stds, mins):
    alphas = (mins - means)/stds
    return tnvar_(alphas) * (stds ** 2)

def tnmv(means, stds, mins): # 448 µs ± 4.93 µs vs. 781 µs ± 2 µs
    alphas = (mins - means)/stds
    aerfcx = torch.special.erfcx(alphas/np.sqrt(2))
    aerf = torch.erf(alphas/np.sqrt(2))
    mean_adjs = torch.max(alphas, np.sqrt(2/np.pi) * torch.where(alphas <= 0, - torch.exp(-(alphas ** 2)/2)/(aerf - 1), 1/aerfcx))
    m2 = torch.sqrt(1 + np.sqrt(2/np.pi) * alphas / aerfcx)
    var_adjs = (m2 - mean_adjs)/(m2 + mean_adjs)
    return torch.sum(means + mean_adjs * stds, dim = -1), torch.sum(var_adjs * (stds ** 2), dim = -1)

def gaussian_cdf(means, variances):
    return (1 + torch.erf(-means/(torch.sqrt(variances) * np.sqrt(2))))/2


# In[4]:


class TruncGaussianPMTensor:
    def __init__(self, means: torch.Tensor, stds: torch.Tensor, mnormxs = None, pm_sizes = None, pm_locs = None, pm_tot_sizes = None, bounded = False):
        self.batch_size = means.shape[0]
        self.tensor_shape = means.shape[1:]
        self.means = means
        self.bounded = bounded
        if isinstance(stds, float):
            self.stds = stds * torch.ones_like(self.means, device = means.device)
        else:
            self.stds = stds
        if mnormxs is None:
            self.mins = torch.zeros_like(self.means, device = means.device)
            self.pm_sizes = torch.tensor([])
            self.pm_locs = torch.tensor([])
            self.pm_tot_sizes = torch.tensor([])
        else: 
            self.mnormxs = mnormxs
            self.pm_sizes = pm_sizes
            self.pm_locs = pm_locs
            self.pm_tot_sizes = pm_tot_sizes

    def __repr__(self):
        return 'N{}({}, {}, {}, {}, {})'.format('b' if self.bounded else '', self.means, self.stds, self.mnormxs, self.pm_sizes, self.pm_locs)
    
    def __getitem__(self, key):
        return TruncGaussianPMTensor(self.means[key: key + 1], self.stds[key: key + 1], self.mnormxs[key: key + 1], self.pm_sizes[key: key + 1], self.pm_locs[key: key + 1], self.pm_tot_sizes[key: key + 1], self.bounded)
    
    def __neg__(self):
        self.mins = - self.mnormxs
        self.maxs = - self.maxs
        self.means = - self.means
        self.pm_locs = - self.pm_locs
    
    def __add__(self, other):
        if isinstance(other, self.__class__):
            raise TypeError(f"Unsupported addition of two '{self.__class__}'")
        elif isinstance(other, (int, float)):
            return TruncGaussianPMTensor(self.means + other, self.stds, self.mnormxs, self.pm_sizes, self.pm_locs + other if len(self.pm_locs) > 0 else self.pm_locs, self.pm_tot_sizes, self.bounded)
        elif isinstance(other, torch.Tensor):
            #assert other.shape == self.tensor_shape
            return TruncGaussianPMTensor(self.means + other, self.stds, self.mnormxs, self.pm_sizes, self.pm_locs + other.view(1, other.shape[0], 1).repeat((self.batch_size, 1, self.pm_locs.shape[2])) if len(self.pm_locs) > 0 else self.pm_locs, self.pm_tot_sizes, self.bounded)
        else:
            raise TypeError(f"Unsupported operand type(s) for +/-: '{self.__class__}' and '{type(other)}'")
    __radd__ = __add__

    def __sub__(self, other):
        return self + -other
    def __rsub__(self, other):
        return -self + other
    
    def __mul__(self, other):
        if isinstance(other, (int, float, torch.Tensor)):
            raise NotImplementedError
            #return TruncGaussianPMTensor(other * self.means, other * self.stds, other * self.mins, self.pm_sizes, other * self.pm_locs, self.pm_tot_sizes)
        else:
            raise TypeError(f"Unsupported operand type(s) for *: '{self.__class__}' and '{type(other)}'")
    __rmul__ = __mul__

    def __matmul__(self, other):
        if isinstance(other, torch.Tensor):
            assert len(self.tensor_shape) == 1
            if torch.all(self.mins == - float("Inf")):
                return TruncGaussianPMTensor(self.means @ other.T, self.stds @ other.T, torch.zeros_like([self.batch_size, other.shape[0]], device = self.mins.device), self.pm_sizes, self.pm_locs, self.pm_tot_sizes, self.bounded)
            else:
                temp_means = torch.cat([(self.means[:,i:i+1] * other[:,i]).unsqueeze(2) for i in range(self.tensor_shape[0])], dim = 2)
                temp_stds = torch.cat([(self.stds[:,i:i+1] * other[:,i]).unsqueeze(2) for i in range(self.tensor_shape[0])], dim = 2)
                other_sign = torch.sign(other)
                temp_mins = torch.cat([(self.mins[:,i:i+1] * torch.where(other_sign[])).unsqueeze(2) for i in range(self.tensor_shape[0])], dim = 2)
                self.temp_means = temp_means
                self.temp_stds = temp_stds
                self.temp_mins = temp_mins
                means, variances = tnmv(temp_means, temp_stds, temp_mins)
                stds = torch.sqrt(variances)
                #print(temp_means.sum(dim = 2)[0,0])
                #print(means[0,0])
                mins = - float("Inf") * torch.ones_like(means)
                # if self.pm_sizes.shape[2] == 1:
                #     print('hi')
                #     return TruncGaussianPMTensor(means, variances, mins, self.pm_sizes, self.pm_locs, self.pm_tot_sizes)
                # else:
                pm_locs = torch.einsum('ijk, lj -> ilkj', self.pm_locs, other).reshape(self.batch_size, means.shape[1], -1)
                pm_sizes = self.pm_sizes.permute(0, 2, 1).unsqueeze(1).repeat(1, means.shape[1], 1, 1).reshape(self.batch_size, means.shape[1], -1)/self.tensor_shape[0]
                pm_tot_sizes = torch.sum(self.pm_tot_sizes, dim = 1, keepdim = True).repeat(1, means.shape[1])/self.tensor_shape[0]
                return TruncGaussianPMTensor(means, stds, mins, pm_sizes, pm_locs, pm_tot_sizes)
                #means = torch.cat([(self.means[:,i:i+1] * other[:,i]).unsqueeze(1) for i in range(self.tensor_shape[0])], dim = 1)
                #variances = torch.cat([(self.variances[:,i:i+1] * (other[:,i] ** 2)).unsqueeze(1) for i in range(self.tensor_shape[0])], dim = 1)
                #mins = torch.cat([(self.mins[:,i:i+1] * other[:,i]).unsqueeze(1) for i in range(self.tensor_shape[0])], dim = 1)
                #return TruncGaussianPMTensor(means, variances, mins, self.pm_sizes, self.pm_locs, self.pm_tot_sizes)

        else:
            raise TypeError(f"Unsupported operand type(s) for *: '{self.__class__}' and '{type(other)}'")
    
    def flatten(self):
        return TruncGaussianPMTensor(self.means.view(self.batch_size, -1), self.stds.view(self.batch_size, -1), self.mins.view(self.batch_size, -1), self.pm_sizes.view(self.batch_size, -1), self.pm_locs.view(self.batch_size, -1), self.tot_sizes)
    
    def relu(self):
        gcdf = gaussian_cdf(self.means, self.stds)
        if len(self.pm_sizes) == 0:
            pm_sizes = gcdf.unsqueeze(2)
            pm_locs = torch.zeros_like(pm_sizes)
            pm_tot_sizes = gcdf
        else:
            pm_pos_sizes = torch.where(self.pm_locs > 0, self.pm_sizes, 0)
            pm_neg_sizes = torch.where(self.pm_locs > 0, 0, self.pm_sizes)
            pm_zero = (torch.sum(pm_neg_sizes, dim = 2) + gcdf * (1 - self.pm_tot_sizes)).unsqueeze(2)
            pm_sizes = torch.cat((pm_pos_sizes, pm_zero), dim = 2)
            pm_locs = torch.cat((self.pm_locs, torch.zeros_like(pm_zero)), dim = 2)
            pm_tot_sizes = self.pm_tot_sizes + gcdf * (1 - self.pm_tot_sizes)
        return TruncGaussianPMTensor(self.means, self.stds, torch.relu(self.mins), pm_sizes, pm_locs, pm_tot_sizes)

    def sample(self, N):
        normals = torch.max(self.mins.unsqueeze(2).expand([*self.means.shape, N]), self.stds.unsqueeze(2).expand([*self.means.shape, N]) * torch.randn([*self.means.shape, N], device = self.means.device) + self.means.unsqueeze(2).expand([*self.means.shape, N]))
        if len(self.pm_sizes) == 0:
            return normals
        if torch.all(self.mins == -float('Inf')):
            fixed_pm_sizes = torch.cat((self.pm_sizes[:,:,:-1], torch.where(self.pm_sizes.sum(dim = 2) == 0, 1, self.pm_sizes[:,:,-1]).unsqueeze(2)), dim = 2)
            pm_tot_sizes = self.pm_tot_sizes
        else:
            if self.pm_sizes.shape[2] == 1:
                pm_tot_sizes = torch.zeros_like(self.pm_tot_sizes)
                fixed_pm_sizes = torch.ones_like(self.pm_sizes)
            else:
                gcdf = gaussian_cdf(self.means, self.variances)
                #pm_tot_sizes = self.pm_tot_sizes
                pm_tot_sizes = (self.pm_tot_sizes - gcdf)/(1-gcdf)
                adj_pm_sizes = torch.cat((self.pm_sizes[:,:,:-1], (self.pm_sizes[:,:,-1] - gcdf * (1 - pm_tot_sizes)).unsqueeze(2)), dim = 2)
                fixed_pm_sizes = torch.cat((adj_pm_sizes[:,:,:-1], torch.where(adj_pm_sizes.sum(dim = 2) == 0, 1, adj_pm_sizes[:,:,-1]).unsqueeze(2)), dim = 2)
        pm_idx = torch.multinomial(fixed_pm_sizes.reshape(-1, self.pm_sizes.shape[2]), N, replacement = True).reshape(self.batch_size, self.tensor_shape[0], -1)
        pms = self.pm_locs[torch.arange(self.batch_size).reshape(self.batch_size, 1, 1).repeat(1, self.tensor_shape[0], N), torch.arange(self.tensor_shape[0]).reshape(1, self.tensor_shape[0], 1).repeat(self.batch_size, 1, N), pm_idx]
        #gpm_ratio = torch.rand([*self.means.shape, N], device = self.means.device)
        #return torch.where(gpm_ratio > pm_tot_sizes.unsqueeze(2).expand([*self.means.shape, N]), normals, pms)
        return torch.sum((self.pm_sizes * self.pm_locs), dim = 2, keepdim = True).expand([*self.means.shape, N]) + normals * (1 - pm_tot_sizes.unsqueeze(2).expand([*self.means.shape, N]))


# In[5]:


x, y = next(iter(train_loader))
x = x.view(-1, 28*28).to(device)


# In[6]:


t = TruncGaussianPMTensor(means = x, variances = 0.1)
t1 = t @ mdl.l1.weight + mdl.l1.bias
t1r = t1.relu()
t2w = t1r @ mdl.l2.weight
t2 = t2w + mdl.l2.bias
t2r = t2.relu()
t3 = t2r @ mdl.l3.weight + mdl.l3.bias


# In[7]:


def sample_mdl(mdl, x, variance, i, j, N = 10000, level = None):
    if level == '1w':
        print(((x + np.sqrt(variance) * torch.randn_like(x)) @ mdl.l1.weight.T).shape)
        return torch.tensor([((x + np.sqrt(variance) * torch.randn_like(x)) @ mdl.l1.weight.T)[i][j] for _ in range(N)])
    elif level == '1':
        return torch.tensor([((x + np.sqrt(variance) * torch.randn_like(x)) @ mdl.l1.weight.T + mdl.l1.bias)[i][j] for _ in range(N)])
    elif level == '1r':
        return torch.tensor([torch.relu((x + np.sqrt(variance) * torch.randn_like(x)) @ mdl.l1.weight.T + mdl.l1.bias)[i][j] for _ in range(N)])
    elif level == '2w':
        return torch.tensor([(torch.relu((x + np.sqrt(variance) * torch.randn_like(x)) @ mdl.l1.weight.T + mdl.l1.bias) @ mdl.l2.weight.T)[i][j] for _ in range(N)])
    elif level == '2':
        return torch.tensor([(torch.relu((x + np.sqrt(variance) * torch.randn_like(x)) @ mdl.l1.weight.T + mdl.l1.bias) @ mdl.l2.weight.T + mdl.l2.bias)[i][j] for _ in range(N)])
    elif level == '2r':
        return torch.tensor([torch.relu(torch.relu((x + np.sqrt(variance) * torch.randn_like(x)) @ mdl.l1.weight.T + mdl.l1.bias) @ mdl.l2.weight.T + mdl.l2.bias)[i][j] for _ in range(N)])
    elif level == '3w':
        return torch.tensor([(torch.relu(torch.relu((x + np.sqrt(variance) * torch.randn_like(x)) @ mdl.l1.weight.T + mdl.l1.bias) @ mdl.l2.weight.T + mdl.l2.bias) @ mdl.l3.weight.T)[i][j] for _ in range(N)])
    else:
        return torch.tensor([mdl(x + np.sqrt(variance) * torch.randn_like(x))[i][j] for i in range(N)])


# In[38]:


t2w_samples = t2w.sample(10000)[0,0].cpu().detach()


# In[39]:


n2w_samples = sample_mdl(x, 0.1, 0, 0, 10000, '2w')


# In[ ]:




