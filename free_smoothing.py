#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


class RectGaussianTensor:
    def __init__(self, means: torch.Tensor, variances: torch.Tensor, mins = None, maxs = None):
        self.batch_size = means.shape[0]
        if mins is None:
            self.num_gaussians = 1
            self.tensor_shape = means.shape[1:]
            self.means = means.unsqueeze(1)
            self.variances = variances.unsqueeze(1)
            self.mins = - torch.ones_like(self.means, device = means.device) * float("Inf")
            self.maxs = torch.ones_like(self.means, device = means.device) * float("Inf")
        else: 
            self.num_gaussians = means.shape[1]
            self.tensor_shape = means.shape[2:]
            self.means = means
            self.variances = variances
            self.mins = mins
            self.maxs = maxs

    def __repr__(self):
        return 'N({}, {}, {}, {})'.format(self.means, self.variances, self.mins, self.maxs)
    
    def __getitem__(self, key):
        return RectGaussianTensor(self.means[key: key + 1], self.variances[key: key + 1], self.mins[key: key + 1], self.maxs[key: key + 1])
    
    def __neg__(self):
        self.mins = - self.mins
        self.maxs = - self.maxs
        self.means = - self.means
    
    def __add__(self, other):
        if isinstance(other, self.__class__):
            return RectGaussianTensor(torch.cat((self.means, other.means)), torch.cat((self.variances, other.variances)), torch.cat((self.mins, other.mins)), torch.cat((self.maxs, other.maxs)))
        elif isinstance(other, (int, float)):
            return RectGaussianTensor(self.means + other, self.variances, self.mins + other, self.maxs + other)
        elif isinstance(other, torch.Tensor):
            assert other.shape == self.tensor_shape
            return RectGaussianTensor(self.means + other, self.variances, self.mins + other, self.maxs + other)
        else:
            raise TypeError(f"Unsupported operand type(s) for +/-: '{self.__class__}' and '{type(other)}'")
    __radd__ = __add__

    def __sub__(self, other):
        return self + -other
    def __rsub__(self, other):
        return -self + other
    
    def __mul__(self, other):
        if isinstance(other, (int, float, torch.Tensor)):
            return RectGaussianTensor(other * self.means, (other ** 2) * self.variances, other * self.mins, other * self.maxs)
        else:
            raise TypeError(f"Unsupported operand type(s) for *: '{self.__class__}' and '{type(other)}'")
    __rmul__ = __mul__

    def __matmul__(self, other):
        if isinstance(other, torch.Tensor):
            assert len(self.tensor_shape) == 1
            return sum([RectGaussianTensor(self.means[:,:,i:i+1] @ other[:,i].T, self.variances[:,:,i:i+1] @ (other[:,i] **2).T, self.mins[:,:,i:i+1] @ other[:,i].T, self.maxs[:,:,i:i+1] @ other[:,i].T) for i in range(self.tensor_shape[0])])
        else:
            raise TypeError(f"Unsupported operand type(s) for *: '{self.__class__}' and '{type(other)}'")
    
    def flatten(self):
        return RectGaussianTensor(self.means.view(self.batch_size, self.num_gaussians, -1), self.variances.view(self.batch_size, self.num_gaussians, -1), self.mins.view(self.batch_size, self.num_gaussians, -1), self.maxs.view(self.batch_size, self.num_gaussians, -1))
    
    def relu(self):
        return RectGaussianTensor(self.means, self.variances, torch.relu(self.mins), torch.relu(self.maxs))


# In[5]:


class TruncGaussianPMTensor:
    def __init__(self, means: torch.Tensor, variances: torch.Tensor, mins = None, pm_sizes = None, pm_locs = None, pm_tot_sizes = None):
        self.batch_size = means.shape[0]
        self.tensor_shape = means.shape[1:]
        self.means = means
        if isinstance(variances, float):
            self.variances = variances * torch.ones_like(self.means, device = means.device)
        else:
            self.variances = variances
        if mins is None:
            self.mins = - torch.ones_like(self.means, device = means.device) * float("Inf")
            self.pm_sizes = torch.tensor([])
            self.pm_locs = torch.tensor([])
            self.pm_tot_sizes = torch.tensor([])
        else: 
            self.mins = mins
            self.pm_sizes = pm_sizes
            self.pm_locs = pm_locs
            self.pm_tot_sizes = pm_tot_sizes

    def __repr__(self):
        return 'N({}, {}, {}, {}, {})'.format(self.means, self.variances, self.mins, self.pm_sizes, self.pm_locs)
    
    def __getitem__(self, key):
        return TruncGaussianPMTensor(self.means[key: key + 1], self.variances[key: key + 1], self.mins[key: key + 1], self.pm_sizes[key: key + 1], self.pm_locs[key: key + 1], self.pm_tot_sizes[key: key + 1])
    
    def __neg__(self):
        self.mins = - self.mins
        self.maxs = - self.maxs
        self.means = - self.means
        self.pm_locs = - self.pm_locs
    
    def __add__(self, other):
        if isinstance(other, self.__class__):
            raise TypeError(f"Unsupported addition of two '{self.__class__}'")
        elif isinstance(other, (int, float)):
            return TruncGaussianPMTensor(self.means + other, self.variances, self.mins + other, self.pm_sizes, self.pm_locs + other if len(self.pm_locs) > 0 else self.pm_locs, self.pm_tot_sizes)
        elif isinstance(other, torch.Tensor):
            #assert other.shape == self.tensor_shape
            return TruncGaussianPMTensor(self.means + other, self.variances, self.mins + other, self.pm_sizes, self.pm_locs + other.view(1, other.shape[0], 1).repeat((self.batch_size, 1, self.pm_locs.shape[2])) if len(self.pm_locs) > 0 else self.pm_locs, self.pm_tot_sizes)
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
            #return TruncGaussianPMTensor(other * self.means, (other ** 2) * self.variances, other * self.mins, self.pm_sizes, other * self.pm_locs, self.pm_tot_sizes)
        else:
            raise TypeError(f"Unsupported operand type(s) for *: '{self.__class__}' and '{type(other)}'")
    __rmul__ = __mul__

    def __matmul__(self, other):
        if isinstance(other, torch.Tensor):
            assert len(self.tensor_shape) == 1
            if torch.all(self.mins == - float("Inf")):
                return TruncGaussianPMTensor(self.means @ other.T, self.variances @ (other ** 2).T, - float("Inf") * torch.ones([self.batch_size, other.shape[0]], device = self.mins.device), self.pm_sizes, self.pm_locs, self.pm_tot_sizes)
            else:
                temp_means = torch.cat([(self.means[:,i:i+1] * other[:,i]).unsqueeze(2) for i in range(self.tensor_shape[0])], dim = 2)
                temp_variances = torch.cat([(self.variances[:,i:i+1] * (other[:,i] ** 2)).unsqueeze(2) for i in range(self.tensor_shape[0])], dim = 2)
                temp_mins = torch.cat([(self.mins[:,i:i+1] * other[:,i]).unsqueeze(2) for i in range(self.tensor_shape[0])], dim = 2)
                self.temp_means = temp_means
                self.temp_variances = temp_variances
                self.temp_mins = temp_mins
                means, variances = tnmv(temp_means, temp_variances, temp_mins)
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
                return TruncGaussianPMTensor(means, variances, mins, pm_sizes, pm_locs, pm_tot_sizes)
                #means = torch.cat([(self.means[:,i:i+1] * other[:,i]).unsqueeze(1) for i in range(self.tensor_shape[0])], dim = 1)
                #variances = torch.cat([(self.variances[:,i:i+1] * (other[:,i] ** 2)).unsqueeze(1) for i in range(self.tensor_shape[0])], dim = 1)
                #mins = torch.cat([(self.mins[:,i:i+1] * other[:,i]).unsqueeze(1) for i in range(self.tensor_shape[0])], dim = 1)
                #return TruncGaussianPMTensor(means, variances, mins, self.pm_sizes, self.pm_locs, self.pm_tot_sizes)

        else:
            raise TypeError(f"Unsupported operand type(s) for *: '{self.__class__}' and '{type(other)}'")
    
    def flatten(self):
        return TruncGaussianPMTensor(self.means.view(self.batch_size, -1), self.variances.view(self.batch_size, -1), self.mins.view(self.batch_size, -1), self.pm_sizes.view(self.batch_size, -1), self.pm_locs.view(self.batch_size, -1), self.tot_sizes)
    
    def relu(self):
        gcdf = gaussian_cdf(self.means, self.variances)
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
        return TruncGaussianPMTensor(self.means, self.variances, torch.relu(self.mins), pm_sizes, pm_locs, pm_tot_sizes)

    def sample(self, N):
        normals = torch.max(self.mins.unsqueeze(2).expand([*self.means.shape, N]), torch.sqrt(self.variances).unsqueeze(2).expand([*self.means.shape, N]) * torch.randn([*self.means.shape, N], device = self.means.device) + self.means.unsqueeze(2).expand([*self.means.shape, N]))
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


# In[6]:


x, y = next(iter(train_loader))
x = x.view(-1, 28*28).to(device)


# In[7]:


t = TruncGaussianPMTensor(means = x, variances = 0.1)
t1 = t @ mdl.l1.weight + mdl.l1.bias
t1r = t1.relu()
t2w = t1r @ mdl.l2.weight
t2 = t2w + mdl.l2.bias
t2r = t2.relu()
t3 = t2r @ mdl.l3.weight + mdl.l3.bias


# In[16]:


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


# In[163]:


t3_samples = t3.sample(10000)[0,0].cpu().detach()


# In[160]:


n3_samples = torch.tensor([mdl(x + np.sqrt(0.1) * torch.randn_like(x))[0][0]  for i in range(10000)])


# In[38]:


t2w_samples = t2w.sample(10000)[0,0].cpu().detach()


# In[39]:


n2w_samples = torch.tensor([(torch.relu((x + np.sqrt(0.1) * torch.randn_like(x)) @ mdl.l1.weight.T + mdl.l1.bias) @ mdl.l2.weight.T)[0,0] for i in range(10000)])


# In[151]:


n2_samples = torch.tensor([(torch.relu((x + np.sqrt(0.1) * torch.randn_like(x)) @ mdl.l1.weight.T + mdl.l1.bias) @ mdl.l2.weight.T + mdl.l2.bias )[0,1] for i in range(10000)])
t2_samples = t2.sample(10000)[0,1].cpu().detach()


# In[90]:


n1_samples = torch.tensor([(torch.relu((x + np.sqrt(0.1) * torch.randn_like(x)) @ mdl.l1.weight.T + mdl.l1.bias) @ mdl.l2.weight.T + mdl.l2.bias )[0,0] for i in range(10000)])
t1_samples = t2.sample(10000)[0,0].cpu().detach()


# In[28]:


t1r.temp_variances.shape, t1r.temp_means.shape, t1r.temp_mins.shape


# In[29]:


test_samples = torch.max(torch.sqrt(t1r.temp_variances).unsqueeze(3).expand([*t1r.temp_variances.shape, 1000]) * torch.randn([*t1r.temp_means.shape, 1000]) + t1r.temp_means.unsqueeze(3).expand([*t1r.temp_variances.shape, 1000]), t1r.temp_mins.unsqueeze(3).expand([*t1r.temp_variances.shape, 1000]))


# In[32]:


t1r.pm_tot_sizes.shape


# In[44]:


gcdf = gaussian_cdf(t1r.means, t1r.variances)


# In[49]:


torch.sum(test_samples, dim = 2)


# In[45]:


test_samples_ = torch.sum(test_samples, dim = 2)[0][0] * (1 - gcdf.unsqueeze(2).expand([*t1r.pm_tot_sizes.shape, 1000]))


# In[46]:


plt.hist(test_samples_.detach()[0][0], bins = 25)


# In[87]:


t2.means[0,0]


# In[170]:


torch.mean(t2_samples), torch.std(t2_samples)


# In[169]:


torch.mean(n2_samples), torch.std(n2_samples)


# In[40]:


plt.hist(t2w_samples, bins = 60);


# In[41]:


plt.hist(n2w_samples, bins = 60);
#plt.hist(t2_samples, bins = 60);


# In[22]:


torch.argmax((((t1r.pm_sizes > 0.3) * (t1r.pm_sizes < 0.4)) + 1)[14,:], dim = 0)


# In[56]:


t1r.pm_sizes[15,73]


# In[66]:


torch.sum(t1r_samples == 0), torch.sum(n1r_samples == 0)


# In[23]:


n1r_samples = sample_mdl(mdl, x, 0.1, 14, 42, 10000, '1r')


# In[24]:


t1r_samples = t1r.sample(10000)[14, 42].cpu().detach()


# In[25]:


plt.hist(n1r_samples, bins = 60);
plt.hist(t1r_samples, bins = 60);


# In[23]:


t1r.means[0,18]


# In[10]:


gaussian_cdf(t1.means[0, 25], t1.variances[0, 25])


# In[120]:


t1r.pm_tot_sizes[0,25]


# In[226]:


ta = torch.tensor([[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]], [[16, 17, 18, 19, 20], [21, 22, 23, 24, 25], [26, 27, 28, 29, 30]]]) # [torch.tensor([[[1, 1, 1, 2, 3], [2, 4, 1, 2, 3], [0, 4, 3, 2, 1]], [[1, 4, 3, 2, 0], [1, 1, 2, 2, 3], [0, 1, 2, 3, 4]]])]


# In[237]:


ta[[[0, 1], [0, 1]], [[1, 2], [0, 1]], [[2, 3], [0, 1]]]


# In[231]:


idx = torch.multinomial(torch.ones((6, 5), dtype = torch.float), num_samples = 10, replacement = True).reshape(2, 3, -1)


# In[240]:


idx = torch.multinomial(torch.ones((640, 3), dtype = torch.float), num_samples = 12, replacement = True).reshape(64, 10, -1)


# In[243]:


torch.rand(64, 10, 100)[torch.arange(64).reshape(64, 1, 1).repeat(1, 10, 12), torch.arange(10).reshape(1, 10, 1).repeat(64, 1, 12), idx]


# In[162]:


a = torch.randn(100, 5, 35)


# In[186]:


a.shape


# In[192]:


a.repeat((100, 1, 35)).shape


# In[165]:


a[a > 0]


# In[161]:


torch.einsum('ijk, lj -> ilkj', torch.randn(10, 512, 33), torch.rand(100, 512)).shape


# In[103]:


torch.sum(t1r[0].sample(10000), dim = 2).argmax()


# In[105]:


t1r_samples


# In[90]:


n1r_samples


# In[116]:


plt.hist(t1r_samples, bins = 20)


# In[37]:


torch.all(t2r.pm_tot_sizes <= 1)


# In[107]:


n1r_samples = torch.tensor([torch.relu((x + np.sqrt(0.1) * torch.randn_like(x)) @ mdl.l1.weight.T + mdl.l1.bias)[0,25] for i in range(10000)])


# In[73]:


t1r_samples.unique()


# In[108]:


n1r_samples.unique()


# In[109]:


plt.hist(n1r_samples, bins = 20)


# In[15]:


t1r.pm_sizes.shape


# In[13]:


t1r_samples = t1r[0].sample(1)


# In[50]:


N = 10000
rs_samples = torch.tensor([mdl(x + np.sqrt(0.1) * torch.randn_like(x))[0][0] for _ in range(N)])


# In[24]:


frs_samples


# In[48]:


sum(t3.sample(10000)[0,0].cpu().detach().numpy() > 0)


# In[45]:


frs_samples = t3.sample(10000)[0,0].cpu().detach().numpy()


# In[53]:


frs_samples.max()


# In[51]:


plt.hist(rs_samples, bins = 20)


# In[46]:


plt.hist(frs_samples, bins = 20)


# In[203]:


t3.pm_sizes.shape


# In[176]:


t1 = t @ mdl.l1.weight + mdl.l1.bias


# In[10]:


t1r = t1.relu()


# In[11]:


t2 = t1r @ mdl.l2.weight + mdl.l2.bias


# In[127]:


torch.sum(torch.max(torch.normal(t2.means[0,:,0], t2.variances[0,:,0]), t2.mins[0,:,0]))/100


# In[12]:


def tnormal(means, variances, N = 10000):
    return torch.sqrt(variances).unsqueeze(3).expand([*means.shape, N]) * torch.randn([*means.shape, N], device = means.device) + means.unsqueeze(3).expand([*means.shape, N])


# In[14]:


samples = tnormal(t2.means.cpu(), t2.variances.cpu())


# In[27]:


samples_clip = torch.sum(torch.max(samples, t2.mins.unsqueeze(3).expand(64, 100, 50, 10000).cpu()), dim = 1).detach().numpy()


# In[23]:


samples.shape


# In[26]:


samples_clip.shape


# In[58]:


res_fake = scipy.stats.normaltest(torch.randn(3200, 10000).numpy(), axis = 1)


# In[59]:


min(res[1])


# In[60]:


res = scipy.stats.normaltest(samples_clip.reshape(-1, 10000), axis = 1)


# In[50]:


scipy.stats.normaltest(samples_clip.reshape(-1, 10000)[1140,:])


# In[61]:


res[1].argmin()


# In[63]:


plt.hist(res_fake[1])


# In[62]:


plt.hist(res[1])


# In[155]:


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


# In[158]:


torch.sum(tnmean(t2.means, torch.sqrt(t2.variances), t2.mins), dim = 1)[0,0]


# In[159]:


a, b = tnmv(t2.means.permute(0, 2, 1), torch.sqrt(t2.variances).permute(0, 2, 1), t2.mins.permute(0, 2, 1))


# In[148]:


a.shape


# In[160]:


a[0,0], b[0,0]


# In[139]:


a.shape, b.shape


# In[156]:


m, v = tnmv(t2.means[0,:,0], torch.sqrt(t2.variances[0,:,0]), t2.mins[0,:,0])


# In[157]:


m, v


# In[134]:


get_ipython().run_cell_magic('timeit', '', 'm, v = torch.sum(tnmean(t2.means[0,:,0], torch.sqrt(t2.variances[0,:,0]), t2.mins[0,:,0])), torch.sum(tnvar(t2.means[0,:,0], torch.sqrt(t2.variances[0,:,0]), t2.mins[0,:,0]))\n')


# In[111]:


m, v = m.cpu().detach().numpy(), v.cpu().detach().numpy()


# In[72]:


torch.sum(t2.means[0,:,0])


# In[123]:


np.var(samples_clip[0, 0, :])


# In[121]:


np.mean(samples_clip[0, 0, :])


# In[119]:


plt.hist(samples_clip.reshape(-1, 10000)[0,:], bins = 50, density = True)
x = np.linspace(m - 3*np.sqrt(v), m + 3*np.sqrt(v), 200)
plt.plot(x, scipy.stats.norm.pdf(x, m, np.sqrt(v)))


# In[211]:


sm.qqplot(samples, line = 'q');


# In[210]:


sm.qqplot(torch.randn(10000), line = 'q');


# In[216]:


scipy.stats.jarque_bera(torch.randn(10000))


# In[217]:


scipy.stats.jarque_bera(samples)


# In[ ]:




