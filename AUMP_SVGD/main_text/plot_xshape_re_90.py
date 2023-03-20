import torch
import torch.autograd as autograd
import autograd.numpy as np
from tqdm import tqdm, trange
import os
import numpy as np
import torch
from src.Tmy_svgd import tmySVGD
import torch.optim as optim
import torch.distributions as D
import pickle
import argparse
import time
from src.rand_mysvgd import min_mySVGD
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
import numpy as np
import torch
import torch.optim as optim
import torch.distributions as D
from tqdm import tqdm, trange
from src.svgd import SVGD
from src.gsvgd import FullGSVGDBatch
from src.kernel import RBF, BatchRBF
from src.utils import plot_particles

from src.manifold import Grassmann
from src.s_svgd import SlicedSVGD
from src.mysvgd import etmySVGD
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
    
    # create multibatch multivariate Gaussian
    cov1 = torch.eye(dim, device=device)
    cov1[:2, :2] = torch.Tensor([[1, correlation], [correlation, 1]])
    cov2 = torch.eye(dim, device=device)
    cov2[:2, :2] = torch.Tensor([[1, -correlation], [-correlation, 1]])
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
parser.add_argument('--nparticles', type=int,default=100, help='no. of particles')
parser.add_argument('--epochs', type=int, default=20000,help='no. of epochs')
parser.add_argument('--nmix', type=int, default=4, help='no. of modes')
parser.add_argument('--metric', type=str, default="energy", help='distance metric')
parser.add_argument('--noise', type=str, default="True", help='whether to add noise')
parser.add_argument('--kernel', type=str, default="rbf", help='kernel')
parser.add_argument('--gpu', type=int, default=4, help='gpu')
parser.add_argument('--seed', type=int, default=235, help='random seed') 
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
eff_dims = [args.effdim] if args.effdim > 0 else [1, 2, 5]
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
    torch.manual_seed(seed)
    list_norm = torch.zeros(10, 4)
    list_gr_var = torch.zeros(10, 4)
    list_tr = torch.zeros(10, 4)
    list_eig = torch.zeros(10, 4)
    list_energy = torch.zeros(10, 4)

    s = -1
    dims = [90, 100]
    for dim in dims:
        print(f"Running for dim: {dim}")
        print("#####################################################")
        s = s+1
        ## target density
        mix_means = torch.zeros((2, dim), device=device)
        mix_means[:, :2] = 1

        distribution = xshaped_gauss_experiment(
            mixture_dist=D.Categorical(torch.ones(mix_means.shape[0], device=device)),
            means=mix_means,
            correlation=correlation
        )


        # sample from target (for computing metric)
        x_target = distribution.sample((nparticles, )).to(device)
        

        # sample from variational density
        x_init =  torch.randn(nparticles, *distribution.event_shape).to(device)

        ## SVGD
        cov = torch.cov(x_target.T)

        if args.method in ["SVGD", "all"]:
            

            print("Running SVGD >>>>>>>>>>>>>>>>>>")
            # sample from variational density
            x = x_init.clone().to(device)
            kernel = Kernel(method="med_heuristic")
            svgd = SVGD(distribution, kernel, optim.Adam([x], lr=lr), device=device)
            
            svgd.fit(x, epochs, verbose=True, save_every=save_every)
            

        theta = x
            
        


        cov_svgd = torch.cov(theta.T)
        print(f"norm_mse of svgd : {torch.linalg.norm(cov - cov_svgd)}")
        list_norm[s,0] = torch.linalg.norm(cov - cov_svgd)
        print(f"mmd of svgd : {comm_func_eval(theta, x_target)['var_dis']}")
        list_gr_var[s,0] = comm_func_eval(theta, x_target)['var_dis']
        print(f"trace of svgd: {torch.trace(cov - cov_svgd)}")
        list_tr[s,0] = torch.trace(cov - cov_svgd)
        (evals, evecs) = torch.linalg.eig(cov - cov_svgd)
        print(f"eig of svgd is : {evals[0]}")
        list_eig[s,0] = evals[0]
        energy  = energy_dis(theta, x_target, dim)
        print(f"energy of svgd is : {energy}")
        list_energy[s, 0] = energy


        print('Running mysvgd >>>>>>>>>>>>>>>>>>>>>>>')
        

        
        x0 = x_init.clone().to(device)
        vector1  = torch.randn(nparticles, dim).to(device)

    
        lr = 0.01
        theta, _ = etmySVGD(kernel,device).update(x0, score, k = 2, n_iter = 6000,   lr= lr, vector=vector1)
            #mean = np.mean(theta, axis=0)  + np.random.random(1)
            #var_theta = np.cov(theta.T) + np.random.random(1)
            #x0 = np.random.multivariate_normal(mean, var_theta,num)
            
            

        cov_mysvgd = torch.cov(theta.T)
        print(f"norm_mse of mysvgd : {torch.linalg.norm(cov - cov_mysvgd)}")
        list_norm[s,1] = torch.linalg.norm(cov - cov_mysvgd)
        print(f"mmd of mysvgd : {comm_func_eval(theta, x_target)['var_dis']}")
        list_gr_var[s,1] = comm_func_eval(theta, x_target)['var_dis']
        print(f"trace of mysvgd: {torch.trace(cov - cov_mysvgd)}")
        list_tr[s,1] = torch.trace(cov - cov_mysvgd)
        (evals, evecs) = torch.linalg.eig(cov - cov_mysvgd)
        print(f"eig of mysvgd is : {evals[0]}")
        list_eig[s,1] = evals[0]
        energy  = energy_dis(theta, x_target, dim)
        print(f"energy of mysvgd is : {energy}")
        list_energy[s, 1] = energy

        if args.method in ["GSVGD", "all"]:
            res_gsvgd = [0] * len(eff_dims)
            def run_gsvgd(eff_dims):
                for i, eff_dim in enumerate(eff_dims):
                    print(f"Running GSVGD with eff dim = {eff_dim}")

                    m = min(20, dim // eff_dim) if args.m is None else args.m
                    print("number of projections:", m)

                    # sample from variational density
                    x_init_gsvgd = x_init.clone()
                    x_gsvgd = x_init_gsvgd.clone()

                    kernel_gsvgd = BatchKernel(method="med_heuristic")
                    optimizer = optim.Adam([x_gsvgd], lr=lr)
                    manifold = Grassmann(dim, eff_dim)
                    U = torch.eye(dim, device=device).requires_grad_(True)
                    U = U[:, :(m*eff_dim)]

                    gsvgd = FullGSVGDBatch(
                        target=distribution,
                        kernel=kernel_gsvgd,
                        manifold=manifold,
                        optimizer=optimizer,
                        delta=delta,
                        T=T,
                        device=device,
                        noise=add_noise
                    )
                    start = time.time()
                    U, metric_gsvgd = gsvgd.fit(x_gsvgd, U, m, epochs, 
                        verbose=True, save_every=save_every, threshold=0.0001*m)
                    elapsed_time = time.time() - start

                    

                    
                return res_gsvgd,x_gsvgd

        res_gsvgd ,x_gsvgd= run_gsvgd(eff_dims)
        theta = x_gsvgd
        cov_gsvgd = torch.cov(theta.T)
        print(f"norm_mse of mysvgd : {torch.linalg.norm(cov - cov_gsvgd)}")
        list_norm[s,2] = torch.linalg.norm(cov - cov_gsvgd)
        print(f"mmd of mysvgd : {comm_func_eval(theta, x_target)['var_dis']}")
        list_gr_var[s,2] = comm_func_eval(theta, x_target)['var_dis']
        print(f"trace of mysvgd: {torch.trace(cov - cov_gsvgd)}")
        list_tr[s,2] = torch.trace(cov - cov_gsvgd)
        (evals, evecs) = torch.linalg.eig(cov - cov_gsvgd)
        print(f"eig of mysvgd is : {evals[0]}")
        list_eig[s,2] = evals[0]
        energy  = energy_dis(theta, x_target, dim)
        print(f"energy of mysvgd is : {energy}")
        list_energy[s, 2] = energy

        if args.method in ["S-SVGD", "all"]:
            # sample from variational density
            print("Running S-SVGD >>>>>>>>>>>>>>>>>>>>>>>")
            x_init_s_svgd = x_init.clone()
            x_s_svgd = x_init_s_svgd.clone().requires_grad_()
            s_svgd = SlicedSVGD(distribution, device=device)

            start = time.time()
            x_s_svgd, metric_s_svgd = s_svgd.fit(
                samples=x_s_svgd, 
                n_epoch=epochs, 
                lr=args.lr_g,
                eps=lr,
                save_every=save_every
            )
        
        theta = x_s_svgd
        cov_ssvgd = torch.cov(theta.T)
        print(f"norm_mse of mysvgd : {torch.linalg.norm(cov - cov_ssvgd)}")
        list_norm[s,3] = torch.linalg.norm(cov - cov_ssvgd)
        print(f"mmd of mysvgd : {comm_func_eval(theta, x_target)['var_dis']}")
        list_gr_var[s,3] = comm_func_eval(theta, x_target)['var_dis']
        print(f"trace of mysvgd: {torch.trace(cov - cov_ssvgd)}")
        list_tr[s,3] = torch.trace(cov - cov_ssvgd)
        (evals, evecs) = torch.linalg.eig(cov - cov_ssvgd)
        print(f"eig of mysvgd is : {evals[0]}")
        list_eig[s,3] = evals[0]
        energy  = energy_dis(theta, x_target, dim)
        print(f"energy of mysvgd is : {energy}")
        list_energy[s, 3] = energy

    torch.save(list_norm, "norm_xshape_mu_vs_90.pt")
    torch.save(list_gr_var, "gr_xshape_mu_vs_90.pt")
    torch.save(list_tr, "tr_xshape_mu_vs_90.pt")
    torch.save(list_eig, "eig_xshape_mu_vs_90.pt")
    torch.save(list_energy, "energy_xshape_mu_vs_90.pt")