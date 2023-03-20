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
from src.Tmy_svgd import etmySVGD
from src.manifold import Grassmann
from src.s_svgd import SlicedSVGD
from src.MP_SVGD import mp_complx_SVGD
import matplotlib.pyplot as plt
from src.rand_mysvgd import mySVGD
from scipy.stats import energy_distance
import pickle
import argparse
import time

import torch.autograd as autograd

class MVN:
    """a multivariate normal distribution"""
    def __init__(self, mu, A):
        self.mu = mu
        self.A = A
        self.inv = torch.linalg.inv(A)
    
    def dlnprob(self, theta):
        return -1*torch.matmul(theta-torch.repeat_interleave(self.mu, theta.shape[0], 0), self.inv)


parser = argparse.ArgumentParser(description='Running xshaped experiment.')
parser.add_argument('--dim', type=int,default=50, help='dimension')
parser.add_argument('--effdim', type=int, default=-1, help='dimension')
parser.add_argument('--lr', type=float,default=0.001, help='learning rate')
parser.add_argument('--lr_g', type=float, default=0.1, help='learning rate for g')
parser.add_argument('--delta', type=float,default=0.01, help='stepsize for projections')
parser.add_argument('--T', type=float, default=1e-4, help='noise multiplier for projections')
parser.add_argument('--nparticles', type=int,default=200, help='no. of particles')
parser.add_argument('--epochs', type=int, default=30000,help='no. of epochs')
parser.add_argument('--nmix', type=int, default=4, help='no. of modes')
parser.add_argument('--metric', type=str, default="energy", help='distance metric')
parser.add_argument('--noise', type=str, default="True", help='whether to add noise')
parser.add_argument('--kernel', type=str, default="rbf", help='kernel')
parser.add_argument('--gpu', type=int, default=3, help='gpu')
parser.add_argument('--seed', type=int, default=0, help='random seed') 
parser.add_argument('--suffix', type=str, default="", help='suffix for res folder')
parser.add_argument('--m', type=int, help='no. of projections')
parser.add_argument('--save_every', type=int, default=200, help='step intervals to save particles')
parser.add_argument('--method', type=str, default="all", help='which method to use')\


args = parser.parse_args([])
device = torch.device(f'cuda:{args.gpu}' if args.gpu != -1 else 'cpu')
dim = args.dim
lr = args.lr
delta = args.delta
T = args.T
nparticles = args.nparticles
epochs = args.epochs
seed = args.seed
eff_dims = 3
nmix = args.nmix
add_noise = True if args.noise == "True" else False
radius = 5
save_every = args.save_every
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
def mix_gauss_experiment(mixture_dist, means):
    '''Mixture of Multivariate gaussian with cov matrices being the identity.
    Args:
        probs: Tensor of shape (nmix,) for the mixture_distribution.
        means: Tensor of shape (nmix, d), where nmix is the number of components 
            and d is the dimension of each component.
    '''
    nmix = means.shape[0]
    comp = D.Independent(D.Normal(means.to(device), torch.ones((nmix, means.shape[1]), device=device)), 1)
    distribution = D.mixture_same_family.MixtureSameFamily(mixture_dist, comp) 
    return distribution


def points_on_circle(theta, rad):
    '''Generate d-dim points whose first two dimensions lies on a circle of 
    radius rad, with position being specified by the angle from the positive 
    x-axis theta.
    '''
    return torch.Tensor([[rad * np.cos(theta + 0.25*np.pi), rad * np.sin(theta + 0.25*np.pi)]])

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
    list_norm = torch.zeros(20, 7)
    list_gr_var = torch.zeros(20, 7)
    list_tr = torch.zeros(20, 7)
    list_eig = torch.zeros(20,7)
    list_energy = torch.zeros(20, 7)


    s = -1
    
    for complx in range(1,50,5):
        print(f"Running for complx: {complx}")
        print("#####################################################")
        s = s+1
        means = torch.zeros(dim, device=device)



        cov = torch.eye(dim, device=device)
        cov[:complx,:complx] = torch.ones((complx,complx), device=device)*0.01
        
        
        
        dig =  torch.diag_embed(torch.diag(cov))
        cov = cov - dig
        #print(cov)
        cov = cov + torch.eye(dim,dim,device=device)

        #print(cov)

        distribution = D.MultivariateNormal(means.to(device), cov)  

            # sample from target (for computing metric)
        x_target = distribution.sample((nparticles, ))
        # sample from variational density
        torch.manual_seed(235)
        x_init = 5 + np.sqrt(2) * torch.randn(nparticles, *distribution.event_shape, device=device)

        model = MVN(means, cov)
        

       

        

        if args.method in ["SVGD", "all"]:
            

            print("Running SVGD >>>>>>>>>>>>>>>>>>")
            # sample from variational density
            x = x_init.clone().detach().to(device)
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
        

        
        x0 = x_init.clone().detach().to(device)
        vector1  = torch.zeros_like(x0).to(device)

        print("Running mySVGD")

        def score(X):
            X_cp = X.clone().detach().requires_grad_()
            log_prob = distribution.log_prob(X_cp)
            score_func = autograd.grad(log_prob.sum(), X_cp)[0]
            return score_func


        # sample from variational density
        
        
        

        
    
       
        lr = 0.01
        theta= etmySVGD(kernel,device,distribution).update(x0,  k = 2, n_iter = 10000,   lr= lr)
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

        x0 = x_init.clone().to(device)
        

        
    
        
        theta= etmySVGD(kernel,device,distribution).update(x0,  k = 3, n_iter = 10000,   lr= lr)
    #mean = np.mean(theta, axis=0)  + np.random.random(1)
    #var_theta = np.cov(theta.T) + np.random.random(1)
    #x0 = np.random.multivariate_normal(mean, var_theta,num)
           
            
            

        cov_mysvgd = torch.cov(theta.T)
        print(f"norm_mse of mysvgd : {torch.linalg.norm(cov - cov_mysvgd)}")
        list_norm[s,5] = torch.linalg.norm(cov - cov_mysvgd)
        print(f"mmd of mysvgd : {comm_func_eval(theta, x_target)['var_dis']}")
        list_gr_var[s,5] = comm_func_eval(theta, x_target)['var_dis']
        print(f"trace of mysvgd: {torch.trace(cov - cov_mysvgd)}")
        list_tr[s,5] = torch.trace(cov - cov_mysvgd)
        (evals, evecs) = torch.linalg.eig(cov - cov_mysvgd)
        print(f"eig of mysvgd is : {evals[0]}")
        list_eig[s,5] = evals[0]
        energy  = energy_dis(theta, x_target, dim)
        print(f"energy of mysvgd is : {energy}")
        list_energy[s, 5] = energy

        x0 = x_init.clone().to(device)
       

        
    
        lr = 0.01
        theta= etmySVGD(kernel,device,distribution).update(x0,  k = 5, n_iter = 10000,   lr= lr)
    
            
            
            

        cov_mysvgd = torch.cov(theta.T)
        print(f"norm_mse of mysvgd : {torch.linalg.norm(cov - cov_mysvgd)}")
        list_norm[s,6] = torch.linalg.norm(cov - cov_mysvgd)
        print(f"mmd of mysvgd : {comm_func_eval(theta, x_target)['var_dis']}")
        list_gr_var[s,6] = comm_func_eval(theta, x_target)['var_dis']
        print(f"trace of mysvgd: {torch.trace(cov - cov_mysvgd)}")
        list_tr[s,6] = torch.trace(cov - cov_mysvgd)
        (evals, evecs) = torch.linalg.eig(cov - cov_mysvgd)
        print(f"eig of mysvgd is : {evals[0]}")
        list_eig[s,6] = evals[0]
        energy  = energy_dis(theta, x_target, dim)
        print(f"energy of mysvgd is : {energy}")
        list_energy[s, 6] = energy


        print('Running MPsvgd >>>>>>>>>>>>>>>>>>>>>>>')
        

        
        x0 = x_init.clone().to(device)
        vector1  = torch.zeros_like(x0).to(device)

        
        x0 = x_init.clone().to(device)
            
        theta = mp_complx_SVGD(kernel, device, distribution).update(x0, n_iter = 8000,  lr=lr,  complx=complx )
            #mean = np.mean(theta, axis=0)  + np.random.random(1)
            #var_theta = np.cov(theta.T) + np.random.random(1)
            #x0 = np.random.multivariate_normal(mean, var_theta,num)
           

        
            
            

        cov_mpsvgd = torch.cov(theta.T)
        print(f"norm_mse of mpsvgd : {torch.linalg.norm(cov - cov_mpsvgd)}")
        list_norm[s,2] = torch.linalg.norm(cov - cov_mpsvgd)
        print(f"mmd of mpsvgd : {comm_func_eval(theta, x_target)['var_dis']}")
        list_gr_var[s,2] = comm_func_eval(theta, x_target)['var_dis']
        print(f"trace of mpsvgd: {torch.trace(cov - cov_mpsvgd)}")
        list_tr[s,2] = torch.trace(cov - cov_mpsvgd)
        (evals, evecs) = torch.linalg.eig(cov - cov_mpsvgd)
        print(f"eig of mpsvgd is : {evals[0]}")
        list_eig[s,2] = evals[0]
        energy  = energy_dis(theta, x_target, dim)
        print(f"energy of mpsvgd is : {energy}")
        list_energy[s, 2] = energy


        if args.method in ["GSVGD", "all"]:
            res_gsvgd = 0
            def run_gsvgd(eff_dims):
                eff_dim = eff_dims
                print(f"Running GSVGD with eff dim = {eff_dim}")

                m = min(20, dim //3) 
                print("number of projections:", m)

                # sample from variational density
                x_init_gsvgd = x_init.clone()
                x_gsvgd = x_init_gsvgd.clone()

                kernel_gsvgd = BatchKernel(method="med_heuristic")
                optimizer = optim.Adam([x_gsvgd], lr=lr)
                manifold = Grassmann(dim, 3)
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
                
                U, metric_gsvgd = gsvgd.fit(x_gsvgd, U, m, 20000, 
                    verbose=True, save_every=save_every, threshold=0.0001*m)
               

                    

                    
                return res_gsvgd,x_gsvgd

        res_gsvgd ,x_gsvgd= run_gsvgd(eff_dims)
        theta = x_gsvgd
        cov_gsvgd = torch.cov(theta.T)
        print(f"norm_mse of gsvgd : {torch.linalg.norm(cov - cov_gsvgd)}")
        list_norm[s,3] = torch.linalg.norm(cov - cov_gsvgd)
        print(f"mmd of gsvgd : {comm_func_eval(theta, x_target)['var_dis']}")
        list_gr_var[s,3] = comm_func_eval(theta, x_target)['var_dis']
        print(f"trace of gsvgd: {torch.trace(cov - cov_gsvgd)}")
        list_tr[s,3] = torch.trace(cov - cov_gsvgd)
        (evals, evecs) = torch.linalg.eig(cov - cov_gsvgd)
        print(f"eig of gsvgd is : {evals[0]}")
        list_eig[s,3] = evals[0]
        energy  = energy_dis(theta, x_target, dim)
        print(f"energy of gsvgd is : {energy}")
        list_energy[s, 3] = energy

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
        print(f"norm_mse of ssvgd : {torch.linalg.norm(cov - cov_ssvgd)}")
        list_norm[s,4] = torch.linalg.norm(cov - cov_ssvgd)
        print(f"mmd of ssvgd : {comm_func_eval(theta, x_target)['var_dis']}")
        list_gr_var[s,4] = comm_func_eval(theta, x_target)['var_dis']
        print(f"trace of ssvgd: {torch.trace(cov - cov_ssvgd)}")
        list_tr[s,4] = torch.trace(cov - cov_ssvgd)
        (evals, evecs) = torch.linalg.eig(cov - cov_ssvgd)
        print(f"eig of ssvgd is : {evals[0]}")
        list_eig[s,4] = evals[0]
        energy  = energy_dis(theta, x_target, dim)
        print(f"energy of ssvgd is : {energy}")
        list_energy[s, 4] = energy

    torch.save(list_norm, "norm_gauss_complx.pt")
    torch.save(list_gr_var, "gr_gauss_complx.pt")
    torch.save(list_tr, "tr_gauss_complx.pt")
    torch.save(list_eig, "eig_gauss_complx.pt")
    torch.save(list_energy, "energy_gauss_complx.pt")