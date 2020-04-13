import math
import torch
import numbers
import sklearn
import numpy as np
from torch import nn
from tqdm import tqdm
import torch.nn.functional as F
from torch.autograd import Variable
from skimage.transform import rotate
from functions.visualization import argmax_ch

def dice_loss(pred, target, ign_first_ch=True):
    eps = 1
    assert pred.size() == target.size(), 'Input and target are different dim'
    
    if len(target.size())==4:
        n,c,x,y = target.size()
    if len(target.size())==5:
        n,c,x,y,z = target.size()

    target = target.view(n,c,-1)
    pred = pred.view(n,c,-1)
    
    if ign_first_ch:
        target = target[:,1:,:]
        pred = pred[:,1:,:]
 
    num = torch.sum(2*(target*pred),2) + eps
    den = (pred*pred).sum(2) + (target*target).sum(2) + eps
    dice_loss = 1-num/den
    ind_avg = dice_loss
    total_avg = torch.mean(dice_loss)
    regions_avg = torch.mean(dice_loss, 0)
    
    return total_avg, regions_avg, ind_avg

def max_bin(input):
    input = input.detach()
    batch_n, chs, xdim, ydim, zdim = input.size()
    m, _  = torch.max(input,1)
    max_mask = []
    for c in range(chs):
        max_mask += [(input[:,c,:,:,:] == m).unsqueeze(1)]
    max_mask = torch.cat(max_mask, 1)
    return max_mask.float()

def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape).cuda()
    return -Variable(torch.log(-torch.log(U + eps) + eps))

def gumbel_softmax_sample(logits, temperature):
    y = logits + sample_gumbel(logits.size())
    return F.softmax(y / temperature, dim=-1)

def gumbel_softmax(logits, temperature):
    """
    ST-gumple-softmax
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logits, temperature)
    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    y_hard = (y_hard - y).detach() + y
    return y_hard

def st_softmax(logits):
    """
    ST-gumple-softmax
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """
    y = F.softmax(logits, dim=-1)
    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    y_hard = (y_hard - y).detach() + y
    return y_hard
    
def padder(input, kernel_size):
    '''
    Use to pad input. 
    Mainly to produce the same size output after gaussian smoothing
    '''
    assert input.dtype == torch.float32, 'Input must be torch.float32'
    assert kernel_size>1, 'Gaussian-smoothing kernel must be greater than 1'
    
    p = (kernel_size+1)//2
    r = kernel_size%2 

    if r>0:
        input = F.pad(input, (p-r,p-r,p-r,p-r,p-r,p-r), mode='replicate')
    elif r==0:
        input = F.pad(input, (p-1, p, p-1, p, p-1, p), mode='replicate')
    return input

def normalize_dim1(x):
    '''
    Ensure that dim1 sums up to zero for proper probabilistic interpretation
    '''
    normalizer = torch.sum(x, dim=1, keepdim=True)
    return x/normalizer

class Sampler():
    '''Sample idx without replacement'''
    def __init__(self, idx):
        self.idx = idx 
        self.iterator = iter(sklearn.utils.shuffle(idx))

    def sequential(self):
        try:
            return next(self.iterator)
        except:
            self.iterator = iter(sklearn.utils.shuffle(self.idx))
            return next(self.iterator)
        
    def shuffle(self):
        self.iterator = iter(sklearn.utils.shuffle(self.idx))