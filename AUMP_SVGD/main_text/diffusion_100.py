import os
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from tqdm import trange
from src.svgd import SVGD
from src.gsvgd import FullGSVGDBatch
from src.kernel import RBF, BatchRBF
from src.manifold import Grassmann
import torch.autograd as autograd
from src.s_svgd import SlicedSVGD
from src.diffusion import Diffusion
from src.rand_mysvgd import mySVGD
import pickle
import argparse
import time
import numpyro
from numpyro.infer import MCMC, NUTS
import numpyro.distributions as npr_dist
import jax.random as random
import jax.numpy as jnp
import jax
from src.Tmy_svgd import etmySVGD

parser = argparse.ArgumentParser(description="Running xshaped experiment.")
parser.add_argument("--dim", type=int, default=30, help="dimension")
parser.add_argument("--effdim", type=int, default=5, help="dimension")
parser.add_argument("--lr", type=float, default=0.0001, help="learning rate")
parser.add_argument("--lr_g", type=float, default=0.0001, help="learning rate for S-SVGD")
parser.add_argument(
    "--delta", type=float, default=0.1, help="stepsize for projections"
)
parser.add_argument(
    "--T", type=float, default=1e-4, help="noise multiplier for projections"
)
parser.add_argument("--nparticles", type=int, default=500, help="no. of particles")
parser.add_argument("--epochs", type=int, default=15000, help="no. of epochs")
parser.add_argument("--noise", type=str, default="True", help="whether to add noise")
parser.add_argument("--kernel", type=str, default="rbf", help="kernel")
parser.add_argument("--gpu", type=int, default=7, help="gpu")
parser.add_argument("--seed", type=int, default=235, help="random seed")
parser.add_argument("--suffix", type=str, default="", help="suffix for res folder")
parser.add_argument("--method", type=str, default="svgd", help="svgd, gsvgd or s-svgd")
parser.add_argument("--save_every", type=int, default=100, help="batch size")

args = parser.parse_args([])

dim = args.dim
lr = args.lr
lr_g = args.lr_g
delta = args.delta
T = args.T
nparticles = args.nparticles
epochs = args.epochs
seed = args.seed
eff_dims = 10
add_noise = True if args.noise == "True" else False
save_every = args.save_every  # save metric values every 100 epochs
print(f"Running for lr: {lr}, nparticles: {nparticles}")

device = torch.device(f"cuda:{args.gpu}" if args.gpu != -1 else "cpu")



dim = args.dim
lr = args.lr
lr_g = args.lr_g
delta = args.delta
T = args.T
nparticles = args.nparticles
epochs = args.epochs

eff_dims = 10
add_noise = True if args.noise == "True" else False
save_every = args.save_every  # save metric values every 100 epochs
print(f"Running for lr: {lr}, nparticles: {nparticles}")

device = torch.device(f"cuda:{args.gpu}" if args.gpu != -1 else "cpu")

results_folder = f"/home/zhoujk/SVGD/SVGD_code/GSVGD-main/res/diffusion{args.suffix}/{args.kernel}_epoch{epochs}_lr{lr}_delta{delta}_n{nparticles}_dim{dim}"
results_folder = f"{results_folder}/seed{seed}"
print(results_folder)
if not os.path.exists(results_folder):
    os.makedirs(results_folder)

if args.kernel == "rbf":
    Kernel = RBF
    BatchKernel = BatchRBF
    
if __name__ == "__main__":

    print(f"Device: {device}")
    

    ## initialize conditional diffusion
    beta = 5
    sigma = 0.1

    distribution = Diffusion(dim, beta, device=device)

    loc = torch.arange(0, dim+1, 1)[1:]
    distribution.loc = loc
    noise_covariance = torch.diag(sigma**2 * torch.ones(len(loc), device=device))
    distribution.set_noise_covariance(noise_covariance)
    distribution.set_C()

    x_true = distribution.brownian_motion((1, dim))
    u_true = distribution.solution(x_true)

    obs_noise = torch.normal(0, sigma, size=(1, len(loc))).to(device)
    obs = u_true[:, loc] + obs_noise
    distribution.obs = obs

    # initialize particles
    x0 = distribution.brownian_motion((nparticles, dim))

    C = distribution.C.cpu().numpy()
    dt = distribution.dt.cpu().numpy()
    loc = distribution.loc.cpu().numpy()
    beta = distribution.beta

        ## SVGD
    ## SVGD
    print("Running SVGD")
    x = x0.clone().requires_grad_()
    # sample from variational density
    kernel = Kernel(method="med_heuristic")

    particles_svgd = torch.zeros([10,500,30], device=device)
    for j in range(10):
        x0 = distribution.brownian_motion((nparticles, dim))
    ## SVGD
        print(f"Running SVGD: eporch : {j}")
        x = x0.clone().requires_grad_()
        # sample from variational density
        kernel = Kernel(method="med_heuristic")
        svgd = SVGD(distribution, kernel, optim.Adam([x], lr=lr), device=device)

        start = time.time()
        svgd.fit(x, epochs=epochs, save_every=save_every)
        elapsed_time = time.time() - start

        fitted_method = svgd
        #particles = fitted_method.particles
        particles_svgd[j,:,:] = x

    



    ## save results
    pickle.dump(
        {
                **{"x_true": x_true},
                **{"u_true": u_true},
                **{"particles": particles_svgd},
                **{"time": elapsed_time}
        },
        open(results_folder + f"/particles_svgd_test.p", "wb")
    )

    # target distribution
    torch.save(distribution, results_folder + '/target_dist_test.p')

    print("Running min_mySVGD")



    # sample from variational density
    particles_mysvgd = torch.zeros([10,500,30], device=device)
    for j in range(10):
        print(f"Running min_mySVGD_2, eporch: {j}")
        x0 = distribution.brownian_motion((nparticles, dim))


        x = x0.clone().requires_grad_()
    # sample from variational density

        lr = 0.001
        theta = etmySVGD(kernel,device,distribution).update(x0,  k = 2, n_iter = 10000,   lr= lr)
        

        particles_mysvgd[j,:,:] = theta

    ## save results
    pickle.dump(
            {
                **{"x_true": x_true},
                **{"u_true": u_true},
                **{"particles": particles_mysvgd},
                **{"time": elapsed_time}
            },
    open(results_folder + f"/particles_AUmp_svgd_2_test.p", "wb")
    )


    print(particles_mysvgd)

    particles_mysvgd = torch.zeros([10,500,30], device=device)
    for j in range(10):
        print(f"Running min_mySVGD_3: eporch{j}")
        x0 = distribution.brownian_motion((nparticles, dim))


        x = x0.clone().requires_grad_()
    # sample from variational density

        lr = 0.001
        theta = etmySVGD(kernel,device,distribution).update(x0,  k = 3, n_iter = 10000,   lr= lr)
        

        particles_mysvgd[j,:,:] = theta

    ## save results
    pickle.dump(
            {
                **{"x_true": x_true},
                **{"u_true": u_true},
                **{"particles": particles_mysvgd},
                **{"time": elapsed_time}
            },
    open(results_folder + f"/particles_AUmp_svgd_3_test.p", "wb")
    )

    particles_mysvgd = torch.zeros([10,500,30], device=device)
    for j in range(10):
        print(f"Running min_mySVGD_5, eporch:{j}")
        x0 = distribution.brownian_motion((nparticles, dim))


        x = x0.clone().requires_grad_()
    # sample from variational density

        lr = 0.001
        theta = etmySVGD(kernel,device,distribution).update(x0,  k = 5, n_iter = 10000,   lr= lr)
        

        particles_mysvgd[j,:,:] = theta

    ## save results
    pickle.dump(
            {
                **{"x_true": x_true},
                **{"u_true": u_true},
                **{"particles": particles_mysvgd},
                **{"time": elapsed_time}
            },
    open(results_folder + f"/particles_AUmp_svgd_5_test.p", "wb")
    )


    particles_gsvgd = torch.zeros([10,500,30], device=device)
    for j in range(10):
        eff_dim = args.effdim
        x0 = distribution.brownian_motion((nparticles, dim))
        print(f"Running GSVGD  :eporch {j}")

        m = min(20, dim // eff_dim)
        print("number of projections:", m)

        # sample from variational density
        x_gsvgd = x0.clone().requires_grad_()

        kernel_gsvgd = BatchKernel(method="med_heuristic")
        optimizer = optim.Adam([x_gsvgd], lr=lr)
        manifold = Grassmann(dim, eff_dim)
        U = torch.eye(dim).requires_grad_(True).to(device)
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
            verbose=True, save_every=save_every, threshold=1e-2)
        elapsed_time = time.time() - start

        particles_gsvgd[j,:,:] = x_gsvgd

    pickle.dump(
    {
                **{"x_true": x_true},
                **{"u_true": u_true},
                **{"particles": particles_gsvgd},
                **{"time": elapsed_time}
            },
    open(results_folder + f"/particles_gsvgd_test.p", "wb")
    )

        ## S-SVGD
    # sample from variational density
    

    particles_ssvgd = torch.zeros([10,500,30], device=device)
    for j in range(10):
        print(f"Running S-SVGD: eporch {j}")
        x0 = distribution.brownian_motion((nparticles, dim))
        x_s_svgd = x0.clone().requires_grad_()
        s_svgd = SlicedSVGD(distribution, device=device)

        start = time.time()
        x_s_svgd, metric_s_svgd = s_svgd.fit(
            samples=x_s_svgd, 
            n_epoch=epochs, 
            lr=lr_g,
            eps=lr,
            save_every=save_every
        )
        elapsed_time = time.time() - start

        fitted_method = s_svgd
        particles_ssvgd[j,:,:] = x_s_svgd

    pickle.dump(
    {
                **{"x_true": x_true},
                **{"u_true": u_true},
                **{"particles": particles_ssvgd},
                **{"time": elapsed_time}
            },
    open(results_folder + f"/particles_ssvgd_test.p", "wb")
    )


