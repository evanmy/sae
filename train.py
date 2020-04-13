'''Input Libraries'''
import os
import torch
import sklearn
import numpy as np
import torch.nn.functional as F
import torch.utils.data as data_utils

from tqdm import tqdm
from functions import mrf as mrf
from functions import models as m
from functions import dataset as data
from functions import training_tools as tt
from functions.visualization import argmax_ch
from functions.parser import train_parser

def run(loader,
        template,
        models,
        optimizer,
        epoch,
        mode,
        args,
        PATH,
        subsample_loader,
        compute_dice=False):
    
    """Define Variables"""
    eps = 1e-12
    u1 = models[0]
    u2 = models[1]
    
    dataset = loader[0]
    sampler = loader[1]
        
    running_mse = []
    running_loss = []
    running_dice = []
    running_consistent = []
    running_prior_loss = []
    running_recon_loss = []
    running_recon_weight = []
    
    """Choose samples"""
    if mode == 'train':
        u1.train()
        u2.train()
        suffix = '_train'

    else:
        u1.eval()
        u2.eval()
        suffix = '_eval'    
        sampler.shuffle()   
        
    running_var = []
    if args.sigma == 2 and mode == 'train':
        if epoch > 0:
            running_var += [args.var] #if args.sigma == 2 else None 

    """Train"""
    for j, si in enumerate(range(subsample_loader)):        

        """Load Data"""
        sample_idx = sampler.sequential()
        inputs, _, target_regs, ids  = dataset[sample_idx]
        prior = template
        
        og_prior = prior.detach().cpu()
        x = inputs.float().cuda().requires_grad_()
        target = target_regs.float().cuda()
        prior = prior.float().cuda().detach()
        log_prior = torch.log(tt.normalize_dim1(prior+eps)).detach()

        """Predict"""
        optimizer.zero_grad()
        
        out = u1(x)
        out = m.enforcer(prior, out)
        n_batch, chs, dim1, dim2, dim3 = out.size()
        logits = out
        out = out.permute(0,2,3,4,1)
        out = out.view(n_batch, dim1*dim2*dim3, chs)
        pred = tt.gumbel_softmax(out, args.tau)
        pred = pred.view(n_batch, dim1, dim2, dim3, chs)
        pred = pred.permute(0,4,1,2,3)         
        
        """Recon"""
        if args.sigma == 0:
            recon = u2(pred)
            # Variance as Hyperparameter
            mse = (recon-x.detach())**2  #mse
            alpha = args.alpha
            mse = torch.sum(mse,(1,2,3,4))    #mse over all dims
            mse = mse.mean()                  #avarage over all batches
            _mse = mse
            recon_loss = alpha*mse 
            running_recon_weight += [alpha]

        elif args.sigma == 2:
            # Estimated Variance
            recon = u2(pred)
            mse = (recon-x.detach())**2
            _mse = (torch.sum(mse.detach(),(1,2,3,4))).mean()

            running_var += [mse.detach().mean().item()]
            rounded_var = 10**np.round(np.log10(args.var))
            running_recon_weight += [np.clip(0.5*(1/(rounded_var)),0, 500)]

            # Weight Reconstruction loss
            mse = np.clip(0.5*(1/(rounded_var)),0, 500)*mse
            mse = torch.sum(mse,(1,2,3,4))    #mse over all dims
            mse = mse.mean()                  #avarage over all batches

            # Since args.var is a scalar now, we need to account for
            # the fact that we doing log det of a matrix
            # Therefore, we multiply by the dimension of the image

            c = dim1*dim2*dim3 #chs is 1 for image

            _var = torch.from_numpy(np.array(args.var+eps)).float()
            recon_loss = mse + 0.5*c*torch.log(_var)
            
        else:
            raise Exception('Not implemented')
                
        """KL Divergence"""
        log_pi = F.log_softmax(logits, 1)
        pi = torch.exp(log_pi)
        
        cce = -1*torch.sum(pi*log_prior,1)      #cross entropy
        cce = torch.sum(cce,(1,2,3))            #cce over all the dims
        cce = cce.mean()               
            
        h = -1*torch.sum(pi*log_pi,1)
        h = torch.sum(h,(1,2,3))
        h = h.mean()
 
        prior_loss = cce - h
    
        """Spatial Consistency"""
        if not args.beta == 0: 
            consistent = args.beta*mrf.spatial_consistency(input= pi, 
                                                           table= args.lookup, 
                                                           neighboor_size= args.k)
        else:
            consistent = torch.zeros(1).cuda()
            
        """Total"""
        loss =  prior_loss + recon_loss + consistent

        if mode=='train':
            loss.backward()
            optimizer.step()
            
        _pred = (argmax_ch(pi)).float().cuda()
        _target_regs = target.float().cuda()
        
        """Overall dice"""
        with torch.no_grad():
            if compute_dice: 
                dice, _, _ = tt.dice_loss(_pred[:,:-1,:,:,:].detach(),
                                          _target_regs[:,:-1,:,:,:].detach(),
                                          ign_first_ch=True)
                running_dice += [(1-dice).item()]
                
                _target_regs = 0
                dice = 0
                _ = 0
            else:
                running_dice += [0]

        """Save"""
        running_consistent += [consistent.item()] 
        running_recon_loss += [recon_loss.item()]
        running_prior_loss += [prior_loss.item()]
        running_loss += [loss.item()]
        running_mse += [_mse.item()]
        
    consistent = np.mean(running_consistent)
    recon_loss = np.mean(running_recon_loss)
    prior_loss = np.mean(running_prior_loss)
    mse = np.mean(running_mse) 
    loss = np.mean(running_loss)
    dice = np.mean(running_dice)
    recon_weight = np.mean(running_recon_weight)
    
    stat = [recon_loss, prior_loss, loss, dice, consistent, mse, recon_weight]
    
    if args.sigma == 2 and mode == 'train': 
        # Average running variance
        args.var = np.mean(running_var)

    if PATH is not None:
        vis.view_results(target_brain= x,
                         recon_brain= recon,
                         pred_regs= _pred,
                         target_regs= target,
                         epoch= epoch,
                         suffix= suffix,
                         show_brain= False,
                         image_idx=0,
                         PATH=PATH)
        
    return stat

if __name__ == "__main__":
    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt
    from functions import visualization as vis

    args = train_parser()
    assert (args.batch_size == 1), "For computational and memory purpose, it currently only support batch==1"
    
    """Select GPU"""
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]= args.gpus
            
    """Load Data"""
    vols_path = './data/vols/'
    aseg_path = './data/labels/'
    train_set = data.load_bucker_data(vols_path,
                                      aseg_path)
    
    """Choose template"""    
    atlas_path = './data/prob_atlas.npz'
    template = data.get_prob_atlas(atlas_path)
    chs = template.shape[1]
    dim1 = template.shape[2]
    dim2 = template.shape[3]
    dim3 = template.shape[4]

    if not args.beta == 0:
        args.lookup = mrf.get_lookup(argmax_ch(template), 
                                     neighboor_size= args.k)        
            
    """Getting idx"""
    indices = len(train_set)
    args.train_idx = np.arange(indices)
    train_sampler = tt.Sampler(args.train_idx)
    
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

    # Decoder
    u2 = m.Simple_Decoder(chs, 1)
    u2 = torch.nn.DataParallel(u2)
    u2.cuda()
    
    params = list(u1.parameters()) + list(u2.parameters())

    optimizer = torch.optim.Adam(params,
                                 lr= args.lr)
    
    """Pretrained Model"""
    # In order obtain good initialization, the encoder was pretrained by
    # mapping the training data to the probabilistic template
    print('============ Loading pretrained weight for enc and dec ============')
    summary = torch.load('./weights/pretrained_encoder.pth.tar')                        
    u1.load_state_dict(summary['u1']) 
                
    """Make folder for save files"""        
    name = 'unsupervised'  
    arguments = ('_sgm'+str(args.sigma)+
                 '_alpha'+str(args.alpha)+
                 '_lr'+str(args.lr)+ 
                 '_beta'+str(args.beta))
     
    PATH = './savepoints/'+name+arguments+'/'

    if not os.path.exists(PATH):
        os.makedirs(PATH)
    else:
        raise Exception('Path exists :(')
    
    if args.sigma == 2:
        args.var = 1e8
      
    """Train"""
    best = 1e8
    train_const = []
    train_recon = []
    train_prior = []
    train_loss = []
    train_dice = []
    train_mse = []
    train_recon_weight = []
    
    for epoch in tqdm(range(args.epochs)):
        mode = 'train'
        stat = run(loader= [train_set, train_sampler],
                   template= template,
                   models= [u1, u2],
                   optimizer= optimizer,
                   epoch= epoch,
                   mode= mode,
                   args= args,
                   PATH= PATH,
                   subsample_loader= len(train_set),
                   compute_dice= args.compute_dice)

        train_recon += [stat[0]]
        train_prior += [stat[1]]
        train_loss += [stat[2]]
        train_dice += [stat[3]]
        train_const += [stat[4]]
        train_mse += [stat[5]]
        train_recon_weight += [stat[6]]
        
        print('UD Epoch %d' % (epoch))
        print('[Train Stat] Recon %.5f Prior: %.5f Total: %.5f Consistency: %.5f Dice: %.3f MSE: % .3f Recon Weight: %.1f'
              % (train_recon[-1],
                 train_prior[-1],
                 train_loss[-1],
                 train_const[-1],
                 train_dice[-1],
                 train_mse[-1], 
                 train_recon_weight[-1]))
        
        train_summary = np.vstack((train_recon,
                                   train_prior,
                                   train_loss,
                                   train_const,
                                   train_dice,
                                   train_mse, 
                                   train_recon_weight))

        """Save model"""
        # Using args.save_last always save the last epoch
        # Otherwise only save the loss improves
        
        state = {'epoch': epoch,
                 'args': args,
                 'u1': u1.state_dict(),
                 'u2': u2.state_dict(),
                 'optimizer' : optimizer.state_dict(),
                 'train_summary': train_summary}        
                
        if train_loss[-1]<best:
            best = train_loss[-1]
            torch.save(state, PATH+'best_model.pth.tar')
        
        if (epoch+1)%50 == 0:
            # Save every 50 epochs
            torch.save(state, PATH+'epoch{}_model.pth.tar'.format(epoch))