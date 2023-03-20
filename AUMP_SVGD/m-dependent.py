import torch
import torch.autograd as autograd
import autograd.numpy as np
from tqdm import tqdm, trange
import os
import numpy as np
import torch

import torch.optim as optim
import torch.distributions as D
import pickle
import argparse
import time
from src.rand_mysvgd import mySVGD
import matplotlib.pyplot as plt
from src.svgd import SVGD
from src.gsvgd import FullGSVGDBatch
from src.kernel import RBF, BatchRBF
from src.utils import plot_particles

from src.manifold import Grassmann
from src.s_svgd import SlicedSVGD
from src.Sliced_KSD_Clean.Util import *
from src.Sliced_KSD_Clean.Divergence.Def_Divergence import *
from src.Sliced_KSD_Clean.Divergence.Kernel import *
from src.Sliced_KSD_Clean.Divergence.Dataloader import *
from scipy.stats import energy_distance
import os
from src.Tmy_svgd import etmySVGD
import numpy as np
import torch
import torch.optim as optim
import torch.distributions as D
from tqdm import tqdm, trange
from src.MP_SVGD import mp_shape_SVGD
from src.svgd import SVGD
from src.gsvgd import FullGSVGDBatch
from src.kernel import RBF, BatchRBF
from src.utils import plot_particles

from src.manifold import Grassmann
from src.s_svgd import SlicedSVGD

import matplotlib.pyplot as plt

import pickle
import argparse
import time

import torch.autograd as autograd


def xshaped_gauss_experiment(mixture_dist, means, correlation):
    '''Mixture of Multivariate gaussian with cov matrices being the identity.
    Args:
        mixture_dist: torch.distributions.Categorical-like instance for the 
            probability of each component in the mixture.
        means: Tensor of shape (nmix, d), where nmix is the number of components 
            and d is the dimension of each component.
        correlation: Float between 0 and 1 for the magnitude of correlation between
            the first two dims.
    '''
    nmix, dim = means.shape
    nmix = nmix + 1
    
    # create multibatch multivariate Gaussian
    cov1 = torch.eye(dim, device=device)
    cov1[:2, :2] = torch.Tensor([[2, 1.9],[1.9, 2]])
   
    
    #cov1[dim-2:, dim-2:] = torch.Tensor([[1, correlation], [correlation, 1]])
    cov2 = torch.eye(dim, device=device)
    cov2[:2, :2] = torch.Tensor([[2, 0], [0, 2]])
    #cov2[dim-2:, dim-2:] = torch.Tensor([[1, 0], [0, 1]])
   
    mix_cov = torch.stack((cov1, cov2))
   
    comp = D.MultivariateNormal(means.to(device), mix_cov)

    distribution = D.mixture_same_family.MixtureSameFamily(mixture_dist, comp)   
    return(distribution)

parser = argparse.ArgumentParser(description='Running xshaped experiment.')
parser.add_argument('--dim', type=int,default=10, help='dimension')
parser.add_argument('--effdim', type=int, default=-1, help='dimension')
parser.add_argument('--lr', type=float,default=0.01, help='learning rate')
parser.add_argument('--lr_g', type=float, default=0.1, help='learning rate for g')
parser.add_argument('--delta', type=float,default=0.01, help='stepsize for projections')
parser.add_argument('--T', type=float, default=1e-4, help='noise multiplier for projections')
parser.add_argument('--nparticles', type=int,default=5000, help='no. of particles')
parser.add_argument('--epochs', type=int, default=50000,help='no. of epochs')
parser.add_argument('--nmix', type=int, default=4, help='no. of modes')
parser.add_argument('--metric', type=str, default="energy", help='distance metric')
parser.add_argument('--noise', type=str, default="True", help='whether to add noise')
parser.add_argument('--kernel', type=str, default="rbf", help='kernel')
parser.add_argument('--gpu', type=int, default=5, help='gpu')
parser.add_argument('--seed', type=int, default=0, help='random seed') 
parser.add_argument('--suffix', type=str, default="", help='suffix for res folder')
parser.add_argument('--m', type=int, help='no. of projections')
parser.add_argument('--save_every', type=int, default=200, help='step intervals to save particles')
parser.add_argument('--method', type=str, default="all", help='which method to use')\


args = parser.parse_args([])
dim = args.dim
lr = args.lr
delta = args.delta
T = args.T
nparticles = args.nparticles
epochs = args.epochs
seed = args.seed
eff_dims = 3
add_noise = True if args.noise == "True" else False
correlation = 0.95
save_every = args.save_every
device = torch.device(f'cuda:{args.gpu}' if args.gpu != -1 else 'cpu')
print(f"Running for dim: {dim}, lr: {lr}, nparticles: {nparticles}")

def comm_func_eval(samples, ground_truth):

    samples = samples.clone()
    ground_truth = ground_truth.clone()

    def ex():
        f0 = torch.mean(samples, axis=0)
        f1 = torch.mean(ground_truth, axis=0)
        return torch.mean((f0-f1)**2)

    def exsqr():
        f0 = torch.var(samples, axis=0)
        f1 = torch.var(ground_truth, axis=0)
        return torch.mean((f0-f1)**2)


    out = {}
    out['mean_dis'] = ex()
    out['var_dis'] = exsqr()
    return out


def score(X):
        X_cp = X.clone().detach().requires_grad_()
        log_prob = distribution.log_prob(X_cp)
        score_func = autograd.grad(log_prob.sum(), X_cp)[0]
        return score_func

def energy_dis(x, x_target, dim):
    a = 0
    for i in range(x.shape[1]):
        x_dim = x[:,i]
        x_target_dim = x_target[:,i]
        a = a + energy_distance(x_dim.cpu().detach().numpy(), x_target_dim.cpu().detach().numpy())
    
    a = a / dim
    
    return a



if args.kernel == "rbf":
    Kernel = RBF
    BatchKernel = BatchRBF






if __name__ == "__main__":
    print(f"Device: {device}")
    
        
        
       
   # torch.manual_seed(seed)
    

    s = -1

    for nparticles in [3000,5000,7500]:
        for dim in range(10,40,10):
            print(f"Running for dim: {dim}")
            print("#####################################################")
            s = s+1
            ## target density
            print(f"Device: {device}")
            

            ## target density
            mix_means = torch.zeros((2, dim), device=device)
            mix_means[:, :2] = 1

            distribution = xshaped_gauss_experiment(
                mixture_dist=D.Categorical(torch.ones(mix_means.shape[0], device=device)),
                means=mix_means,
                correlation=correlation
            )
            

            # sample from target (for computing metric)
            x_target = distribution.sample((nparticles, ))
            # sample from variational density
            
            x_init = 2 + np.sqrt(2) * torch.randn(nparticles, *distribution.event_shape, device=device)

            cov = torch.cov(x_target.T)
            

        

            

            if args.method in ["SVGD", "all"]:
                

                print("Running SVGD >>>>>>>>>>>>>>>>>>")
                # sample from variational density
                x = x_init.clone().detach().to(device)
                kernel = Kernel(method="med_heuristic")
                svgd = SVGD(distribution, kernel, optim.Adam([x], lr=lr), device=device)
                
                svgd.fit(x, epochs, verbose=True, save_every=save_every)
                

            theta = x
            torch.save(theta,'m_'+str(nparticles)+'_'+str(dim))