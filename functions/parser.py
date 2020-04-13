import torch
import numpy as np
from argparse import ArgumentParser

def train_parser():

    parser = ArgumentParser()

    parser.add_argument("gpus", type=str,
                        help="Which gpus to use?")

    parser.add_argument("sigma",
                        type=int,
                        choices={0, 2},
                        help='0: Variance as hyperparameter 2: Estimated variance')  
    
    parser.add_argument("--lr",
                        type=float,
                        dest="lr",
                        default=1e-3,
                        help="Encoder learning rate, Default: 1e-3")

    parser.add_argument("--epochs",
                        type=int,
                        dest="epochs",
                        default=500,
                        help="Training epochs, Default: 500")

    parser.add_argument("--batch_size",
                        type=int,
                        dest="batch_size",
                        default=1,
                        choices={1},
                        help="Only supports batch of 1 for now")
    
    parser.add_argument("--compute_dice",
                        action='store_true',
                        help='Compute dice during training')

    parser.add_argument("--tau",
                        type=float,
                        dest="tau",
                        default=0.6667,
                        help="Temperature for gumbel. Default 0.6667")
  
    parser.add_argument("--alpha",
                        default=1,
                        type=float,
                        help="Weight on the recon term. Only use when args.sigma==0. Default: 1")

    parser.add_argument("--beta",
                        default=0.01,
                        type=float,
                        help="Spatial consistency weight. Default: 0.01")

    parser.add_argument("--k",
                        default=3,
                        type=int,
                        help="Neighboor size for MRF. Only used when args.beta>0. Default: 3")    
    
    args = parser.parse_args()
    
    return args
