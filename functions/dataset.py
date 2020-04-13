import os
import torch
import numpy as np
import torch.nn.functional as F
import torch.utils.data as data

def get_prob_atlas(path):
    atlas = torch.from_numpy(np.load((path))['vol_data']).float()

    subset_regs = [[0,0],   #Background
                   [13,52], #Pallidum   
                   [18,54], #Amygdala
                   [11,50], #Caudate
                   [3,42],  #Cerebral Cortex
                   [17,53], #Hippocampus
                   [10,49], #Thalamus
                   [12,51], #Putamen
                   [2,41],  #Cerebral WM
                   [8,47],  #Cerebellum Cortex
                   [4,43],  #Lateral Ventricle
                   [7,46],  #Cerebellum WM
                   [16,16]] #Brain-Stem

    atlas_label = np.array([0,2,3,4,5,7,8,10,
                            11,12,13,14,15,16,
                            17,18,24,26,28,29,
                            30,31,41,42,43,44,
                            46,47,49,50,51,52,
                            53,54,58,60,62,63,
                            72,77,80,85])

    dim1, dim2, dim3, _= atlas.shape
    chs = 14
    one_hot = torch.zeros(1, chs, dim1, dim2, dim3)

    for i,s in enumerate(subset_regs):
        if not s[0]==s[1]:
            idx1 = np.argwhere(atlas_label == s[0]).flatten()[0]
            idx2 = np.argwhere(atlas_label == s[1]).flatten()[0]
            one_hot[0,i,:,:,:] = atlas[:,:,:,idx1] + atlas[:,:,:,idx2]
        else:
            idx = np.argwhere(atlas_label == s[0]).flatten()[0]
            one_hot[0,i,:,:,:] = atlas[:,:,:,idx]

    mask = one_hot.sum(1).squeeze()
    mask = torch.clamp(mask, 0, 1) 
    ones = torch.ones_like(mask)
    non_roi = ones-mask  
    one_hot[0,-1,:,:,:] = non_roi    
    assert one_hot.sum(1).sum() == dim1*dim2*dim3, 'One-hot encoding does not added up to 1'
    return one_hot

def get_onehot(asegs):
    subset_regs = [[0,0],   #Background
                   [13,52], #Pallidum   
                   [18,54], #Amygdala
                   [11,50], #Caudate
                   [3,42],  #Cerebral Cortex
                   [17,53], #Hippocampus
                   [10,49], #Thalamus
                   [12,51], #Putamen
                   [2,41],  #Cerebral WM
                   [8,47],  #Cerebellum Cortex
                   [4,43],  #Lateral Ventricle
                   [7,46],  #Cerebellum WM
                   [16,16]] #Brain-Stem

    dim1, dim2, dim3 = asegs.shape
    chs = 14
    one_hot = torch.zeros(1, chs, dim1, dim2, dim3)

    for i,s in enumerate(subset_regs):
        combined_vol = (asegs == s[0]) | (asegs == s[1]) 
        one_hot[:,i,:,:,:] = torch.from_numpy(combined_vol*1).float()

    mask = one_hot.sum(1).squeeze() 
    ones = torch.ones_like(mask)
    non_roi = ones-mask    
    one_hot[0,-1,:,:,:] = non_roi    

    assert one_hot.sum(1).sum() == dim1*dim2*dim3, 'One-hot encoding does not added up to 1'
    return one_hot
    
class load_bucker_data(data.Dataset):
    def __init__(self, vols_path, aseg_path):
        super(load_bucker_data, self).__init__()
        self.vols_path = vols_path
        self.vols_files = os.listdir(vols_path)
        self.aseg_path = aseg_path
        
    def __len__(self):
        return len(self.vols_files)
    
    def __getitem__(self, index):
        
        if type(index)==np.ndarray or type(index)==torch.Tensor:
            assert len(index) == 1, 'Only minibatch of 1 supported'
            index = index[0]        
        
        id = self.vols_files[index][:-4]
        
        mri = np.load(self.vols_path +
                      id + 
                      '.npz')
        mri = mri['vol_data']
        mri = torch.from_numpy(mri).unsqueeze(0).unsqueeze(0)
    
        aseg = np.load(self.aseg_path +
                       id + 
                       '.npz')
        
        aseg = aseg['vol_data'].astype('float32')
        onehot = get_onehot(aseg)
        aseg = torch.from_numpy(aseg)
        
        return mri.float(), aseg, onehot.float(), id