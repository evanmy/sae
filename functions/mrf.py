import torch
import skimage 
import numpy as np
from torch import nn
import torch.nn.functional as F
from functions import training_tools as tt

def get_lookup(prior, neighboor_size):
    '''
    prior = one-hot encoded prior segmentation as torch tensor
            shape should be [1, labels, dim1, dim2, dim3]
    '''
    k = neighboor_size
    assert prior.dtype == torch.uint8, 'The prior should be one-hot encoded byte'
    assert np.any(np.linspace(3,21,10) == k), ('Make sure that the neighboour_size' + 
                                               'is within np.linspace(3,21,10)')
    labels = prior.size(1)
    
    enumerate_chs = torch.arange(labels).view(1, -1, 1, 1, 1).byte()
    enumerated_prior = enumerate_chs*prior
    enumerated_prior = torch.sum(enumerated_prior,1,True) 
    enumerated_prior = tt.padder(enumerated_prior.float(), k)
    enumerated_prior = enumerated_prior.squeeze().int().numpy()
    windows = skimage.util.view_as_windows(enumerated_prior, (k,k,k)) # windows.shape 
                                                                      #(dim1, dim2, dim3, k, k, k)
    centers = windows[:, :, :, k//2, k//2, k//2]

    windows = windows.reshape(-1,k,k,k)
    centers = centers.reshape(-1)

    lookup_table = np.zeros((labels, labels))
    for condition in range(labels):
        idx = (centers == condition)   
        for s in range(labels):
            if condition == s:
                # removing repeated counts that comes from the center
                counts = np.sum(windows[idx] == s) - windows[idx].shape[0]
            else: 
                counts = np.sum(windows[idx] == s)
            lookup_table[condition, s] = counts

    norm = np.sum(lookup_table,
                  axis=1,
                  keepdims=True)
    norm = np.tile(norm, (1, labels))

    lookup_table = lookup_table/norm 

    assert (np.allclose(lookup_table.sum(1), 
                        np.ones_like(lookup_table.sum(1)))), 'Row doesnt add up to 1'
    
    return lookup_table

def neighboor_q(input, neighboor_size):
    '''
    Calculate the product of all q(s|x) around voxel i
    Uses convolution to sum the log(q_y) then takes the exp
    
    input: prob of q
    '''
    
    k = neighboor_size
    assert np.any(np.linspace(3,21,10) == k), ('Make sure that the neighboour_size' + 
                                               'is within np.linspace(3,21,10)')    
    x = tt.padder(input,
                  kernel_size= k)

    chs = x.shape[1]

    filter = torch.ones(k, k, k).view(1, 1, k, k, k)
    filter[:, :, k//2, k//2, k//2] = 0
    filter = filter.repeat(chs,1,1,1,1).float().cuda()
    filter.requires_grad = False

    out = F.conv3d(x, 
                   weight= filter,
                   stride= 1, 
                   groups= chs)
    return out

def spatial_consistency(input, table, neighboor_size):
    '''
    KL divergence between q(s|x) and markov random field
    
    input: prob of q
    table: lookup table as probability. Rows add up to 1
    '''
    eps = 1e-12
    n_batch, chs, dim1, dim2, dim3 = input.shape
    q_i = input 
    q_y = neighboor_q(input, neighboor_size)
    assert q_i.shape == q_y.shape, 'q_y and q_i should be the same shape'
    
    # To log probability table
    assert (np.allclose(table.sum(1), 
                        np.ones_like(table.sum(1)))), 'Row doesnt add up to 1'
    
    M = torch.from_numpy(table+eps).float()
    M = M/torch.sum(M, 1, True) #Normalize to account for the extra eps
    M = torch.log(M).cuda()
    assert M.shape == torch.Size([chs,chs]), 'Table dims dont match number of labels'
    M = M.view(1, chs, chs)
    
    #Multiplication
    q_i = input.view(n_batch, chs, dim1*dim2*dim3) 
    q_y = q_y.view(n_batch, chs, dim1*dim2*dim3)
    out = torch.bmm(M, q_y)  # shape [n_batch, chs, dim1*dim2*dim3]
    out = torch.sum(q_i*out,1)
    return -1*torch.sum(out)