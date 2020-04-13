'''Input Libraries'''
import os
import torch
import numpy as np
import torch.nn.functional as F

from tqdm import tqdm
from functions import mrf as mrf
from functions import models as m
from argparse import ArgumentParser
from functions import dataset as data
from functions import training_tools as tt
from functions.visualization import argmax_ch

def eval_parser():
    parser = ArgumentParser()

    parser.add_argument("gpus", 
                        type=str,
                        help="Which gpus to use?")
    
    parser.add_argument("--saved_path", 
                        type=str,
                        default='./weights/trained_mrf_model.pth.tar',
                        help="Path to the saved model. Default: ./weights/trained_mrf_model.pth.tar")    
       
    return  parser.parse_args()

def run(loader,
        template,
        models,
        optimizer,
        epoch,
        mode,
        args,
        PATH,
        subsample_loader=16,
        compute_dice=False):

    """Define Variables"""
    eps = 1e-12
    u1 = models[0]
    u2 = models[1]

    dataset = loader[0]
    test_idx = loader[1]

    overall = []
    individual = []
    file_names = []    
    
    """Choose samples"""
    u1.eval()
    u2.eval()
    
    """Train"""
    for idx in tqdm(test_idx):
        """Load Data"""
        x, _, target, ids  = dataset[idx]
        x = x.float().cuda().requires_grad_()
        target = target.float().cuda()
        prior = template
        prior = prior.float().cuda().detach()    

        """Predict"""
        out = u1(x)
        out = m.enforcer(prior, out) 
        logits = out
        log_pi = F.log_softmax(logits, 1)
        pi = torch.exp(log_pi)
        recon = u2(pi)
        _pred = (argmax_ch(pi)).float().cuda()
        _target_regs = target.float().cuda()
 
        dice_loss, _, indv_avg = tt.dice_loss(_pred[:,:-1,:,:,:].detach(),
                                              _target_regs[:,:-1,:,:,:].detach(),
                                              ign_first_ch=True)
        overall += [dice_loss.cpu().numpy()]
        individual += [indv_avg.cpu().numpy()]
        file_names += [ids]
            
    individual = np.vstack(individual)
    file_names = np.vstack(file_names)
    overall= np.vstack(overall).reshape(-1,1)
    saved_model = np.concatenate((individual, overall),1)
    return saved_model, file_names

if __name__ == "__main__":
    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt
    from functions import visualization as vis

    eval_args = eval_parser()

    """Select GPU"""
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]= eval_args.gpus
    
    summary = torch.load(eval_args.saved_path)     
        
    """Load Data"""
    vols_path = './data/vols/'
    aseg_path = './data/labels/'
    dataset = data.load_bucker_data(vols_path,
                                    aseg_path)
    """Getting idx"""
    indices = len(dataset)
    test_idx = np.arange(indices)

    """Choose template"""
    # Load Atlas
    atlas_path = './data/prob_atlas.npz'
    template = data.get_prob_atlas(atlas_path)
    chs = template.shape[1]
    dim1 = template.shape[2]
    dim2 = template.shape[3]
    dim3 = template.shape[4]  
    
    """Making Model"""
    enc_nf = [4, 8, 16, 32]
    dec_nf = [32, 16, 8, 4]

    # Encoder
    u1 = m.Simple_Unet(input_ch=1,
                       out_ch=chs,
                       use_bn= False,
                       enc_nf= enc_nf,
                       dec_nf= dec_nf)

    u1 = torch.nn.DataParallel(u1)
    u1.cuda()
    u1.load_state_dict(summary['u1'])
    
    # Decoder
    u2 = m.Simple_Decoder(chs, 1)
    u2 = torch.nn.DataParallel(u2)
    u2.cuda()
    u2.load_state_dict(summary['u2'])
    
    """Evaluate"""
    mode = 'eval'
    with torch.no_grad():
        stat, file_names = run(loader= [dataset, test_idx],
                               template= template,
                               models= [u1, u2],
                               optimizer= None,
                               epoch= None,
                               mode= mode,
                               args= None,
                               PATH= None,
                               subsample_loader= None,
                               compute_dice= True)

    print('Average Dice is {}'.format(1-np.mean(stat[:,-1])))
    state = {'stat': stat}     
    torch.save(state, 'result.pth.tar')   