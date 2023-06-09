import os
import matplotlib.pyplot as plt 
import seaborn as sns
import pickle
import numpy as np
import pandas as pd
import torch
from metrics import Metric
from sdiffusion import Diffusion
import argparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description='Plotting metrics.')
parser.add_argument('--exp', type=str, help='Experiment to run')
parser.add_argument('--root', type=str, default="res", help='Root dir for results')
parser.add_argument('--nparticles', type=int, default=100, help='Num of particles')
parser.add_argument('--dim', type=int, default=100, help='Num of particles')
parser.add_argument('--epochs', type=int, default=1000, help='Num of epochs')
parser.add_argument('--lr_svgd', type=float, default=0.001, help='learning rate')
parser.add_argument('--lr_ssvgd', type=float, default=0.001, help='learning rate')
parser.add_argument('--lr_gsvgd', type=float, default=0.001, help='learning rate')
parser.add_argument('--lr_g', type=float, default=0.01, help='learning rate for g')
parser.add_argument('--delta', type=float, help='stepsize for projections')
parser.add_argument('--noise', type=str, default="True", help='noise')
parser.add_argument('--format', type=str, default="png", help='format of figs')
args = parser.parse_args()
dim = args.dim
nparticles = args.nparticles
lr_svgd = args.lr_svgd
lr_ssvgd = args.lr_ssvgd
lr_gsvgd = args.lr_gsvgd
noise = "_noise" if args.noise=="True" else ""

basedir = f"{args.root}/{args.exp}"
resdir = f"rbf_epoch{args.epochs}_lr{lr_gsvgd}_delta{args.delta}_n{nparticles}_dim{dim}"
resdir_svgd = f"rbf_epoch{args.epochs}_lr{lr_svgd}_delta0.1_n{nparticles}_dim{dim}"
resdir_ssvgd = f"rbf_epoch{args.epochs}_lr{lr_ssvgd}_delta0.1_n{nparticles}_dim{dim}"

seeds = range(20)
eff_dims = [1, 2, 5] + [i for i in range(10, 110, 10)]

if __name__ == "__main__":

  # initialize list to store results
  df_ls = []

  for seed in seeds:
    print(f"loading seed {seed}")
    path = f"{basedir}/{resdir}/seed{seed}"
    path_svgd = f"{basedir}/{resdir_svgd}/seed{seed}"
    path_ssvgd = f"{basedir}/{resdir_ssvgd}/seed{seed}"

    method_ls = []
    method_names = []

    # load results
    svgd_res = pickle.load(open(f"{path_svgd}/particles_svgd.p", "rb"))
    ssvgd_res = pickle.load(open(f"{path_ssvgd}/particles_s-svgd_lrg{args.lr_g}.p", "rb"))
    hmc_res = pickle.load(open(f"{path}/particles_hmc.p", "rb"))
    
    method_ls += [svgd_res, ssvgd_res]
    method_names += ["SVGD", "S-SVGD"]

    for eff_dim in eff_dims:
      gsvgd_res = pickle.load(open(f"{path}/particles_gsvgd_effdim{eff_dim}.p", "rb"))
      method_ls.append(gsvgd_res)
      method_names.append(f"GSVGD{eff_dim}")

    # hmc samples
    hmc_res["particles"] = [torch.Tensor(hmc_res["particles"])]
        
    # load target distribution
    target_dist = torch.load(f"{path}/target_dist.p", map_location=device)
    target_dist.device = device

    subplot_c = 3
    subplot_r = int(np.ceil(len(method_ls) / subplot_c))

    fig = plt.figure(figsize=(subplot_c*5, subplot_r*5))

    ## plot solutions
    # get HMC solutiondevice
    hmc_particles = hmc_res["particles"][-1].to(target_dist.device)
    hmc_sol_particles = target_dist.solution(hmc_particles).cpu().numpy()
    
    # instantiate metric function
    metric_fn = Metric(metric="energy", x_init=None, 
        x_target=torch.Tensor(hmc_sol_particles), target_dist=None)

    df_list_all = []
    df_dict = {
      "eff_dim": [],
      "energy": [],
      "seed": [],
      "method": []
    }
    for i, (res, method_name) in enumerate(zip(method_ls, method_names)):

      df_list = []

      particles = res["particles"][-1].to(target_dist.device)
      sol_particles = target_dist.solution(particles).cpu().numpy()

      true_sol = res["u_true"].cpu().numpy().reshape((-1,))
      
      dim = sol_particles.shape[1]
      nparticles = particles.shape[0]
      
      if "GSVGD" in method_name:
        df_dict["energy"] += [metric_fn(torch.Tensor(sol_particles))]
        df_dict["seed"] += [seed]
        df_dict["method"] += ["GSVGD"]
        df_dict["eff_dim"] += [int(method_name.split("GSVGD")[-1])]
        df_dict["run_time"] = res["time"]
      else:
        df_dict["energy"] += [metric_fn(torch.Tensor(sol_particles))] * len(eff_dims)
        df_dict["seed"] += [seed] * len(eff_dims)
        df_dict["method"] += [method_name] * len(eff_dims)
        df_dict["eff_dim"] += eff_dims
        df_dict["run_time"] = res["time"]

    # store results in dictionary
    new_df = pd.DataFrame(df_dict)
    df_ls.append(new_df)

  
  # append all dataframes
  df = pd.concat(df_ls)

  ## plot
  fig = plt.figure(figsize=(10, 6))
  g = sns.lineplot(
    data=df,
    x="eff_dim", 
    y="energy", 
    hue="method",
    markers=True,
    markersize=18,
    # alpha=1,
    ci="sd"
  )
  plt.xlabel("Projection Dimensions", fontsize=38)
  plt.xticks(fontsize=32)
  plt.ylabel("Energy Distance", fontsize=38)
  plt.yticks(fontsize=32)
  plt.legend(fontsize=28, markerscale=2, bbox_to_anchor=(0.95, 0.7), loc='upper right')
  fig.tight_layout()

  fig.savefig(f"{basedir}/{resdir}/solution_effdim.png")
  fig.savefig(f"{basedir}/{resdir}/solution_effdim.pdf")
