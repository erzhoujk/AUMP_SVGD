import os
import matplotlib.pyplot as plt 
import seaborn as sns
import pickle
import numpy as np
import pandas as pd
import torch
from src.diffusion import Diffusion_plot
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
args = parser.parse_args([])
dim = args.dim
nparticles = args.nparticles
lr_svgd = args.lr_svgd
lr_ssvgd = args.lr_ssvgd
lr_gsvgd = args.lr_gsvgd
noise = "_noise" if args.noise=="True" else ""


resdir = f"/home/zhoujk/SVGD/SVGD_code/GSVGD-main/res/diffusion/rbf_epoch20000_lr0.0001_delta0.1_n2500_dim50/seed0"



seeds = range(1)

if __name__ == "__main__":

  for seed in seeds:
    

    # load results
    svgd_res = pickle.load(open(f"{resdir}/particles_svgd1.p", "rb"))
    ssvgd_res = pickle.load(open(f"{resdir}/particles_ssvgd_test1.p", "rb"))
    gsvgd_res = pickle.load(open(f"{resdir}/particles_AUmp_svgd_10_test1.p", "rb"))
    # hmc_res = pickle.load(open(f"{resdir}/particles_hmc.p", "rb"))
    AUmp_svgd = pickle.load(open(f"{resdir}/particles_AUmp_svgd_2_test1.p", "rb"))
    
    method_ls = [svgd_res, ssvgd_res, gsvgd_res, AUmp_svgd]
    method_names = ["SVGD", "S-SVGD","gsvgd",  "AUmp_svgd"]

  


   
        
    # load target distribution
    target_dist = torch.load(f"{resdir}/target_dist.p", map_location=device)

    subplot_c = 5
    subplot_r = int(np.ceil(len(method_ls) / subplot_c))

    fig = plt.figure(figsize=(subplot_c*6, subplot_r*5))

    ## plot solutions
    df_list_all = []
    for i, (res, method_name) in enumerate(zip(method_ls, method_names)):
        df_list = []

        target_dist.device = device
        particles = res["particles"][-1].to(target_dist.device)
        sol_particles = target_dist.solution(particles).cpu().detach().numpy()

        true_sol = res["u_true"].cpu().detach().numpy().reshape((-1,))
        time_step = np.append(0, target_dist.t.cpu().detach().numpy())
        
        dim = sol_particles.shape[1]
        nparticles = particles.shape[0]
        df_new = pd.DataFrame({
          "particles": sol_particles.reshape((-1,)),
          "sample_ind": np.repeat(range(dim), nparticles),
          "method": method_name,
          "seed": seed,
        })

        # append true samples
        df_true = pd.DataFrame({
          "time": time_step,
          "particles": true_sol,
          "sample_ind": 0,
          "method": "True",
          "seed": seed
        })

        df_new = df_new.append(df_true)

        df_list.append(df_new)
        df_list_all.append(df_new)

        # 90%-CI
        for ind, subdf in enumerate(df_list):
          subdf["lower"] = np.nan
          subdf["upper"] = np.nan
          for t in time_step:
            quantiles = subdf[subdf.time == t].particles.quantile([0.05, 0.95]).to_list()
            subdf.loc[subdf.time == t, "lower"] = quantiles[0]
            subdf.loc[subdf.time == t, "upper"] = quantiles[1]
            subdf.loc[subdf.time == t, "mean"] = subdf[subdf.time == t].particles.mean()
          
          df_list[ind] = subdf
        
        df = pd.concat(df_list)
        df = df.sort_values(["method", "time"]).reset_index(drop=True)

        # plot observations
        plt.subplot(subplot_r, subplot_c, i+1)
        sns.lineplot(
          x=time_step[target_dist.loc.cpu().numpy()],
          y=target_dist.obs.cpu().squeeze().numpy(),
          color="red",
          linestyle="",
          marker="o",
          markersize=18,
        )

        # plot solutions
        g = sns.lineplot(
          data=df, 
          x="time", 
          y="particles", 
          hue="method", 
          style="method", 
          ci=None,
          linewidth=3,
        )

        plt.fill_between(data=df.loc[df.method==method_name], 
          x="time", y1="lower", y2="upper", alpha=0.2)
        plt.title(method_name, fontsize=42)
        plt.xlim(0, 1)
        plt.ylim(-1.75, 0.35)
        plt.xlabel("Time Steps", fontsize=38)
        plt.xticks(fontsize=26)
        plt.ylabel("Solution", fontsize=42)
        plt.yticks(fontsize=26)
        if i < subplot_c:
          g.set(xticks=[])
          plt.xlabel(None)
        if i != 0 and i != subplot_c:
          plt.ylabel(None)
        if i == 0:
          plt.legend(loc="upper center", borderaxespad=0., 
            fontsize=25, labels=["Data", "Mean", "True"])
        else:
          plt.legend([],[], frameon=False)
        fig.tight_layout()

    fig.savefig(f"{resdir}/solution.png")
    fig.savefig(f"{resdir}/solution.pdf")


    df_all = pd.concat(df_list_all)
    df_all = df_all.loc[(df_all.method == "HMC") | (df_all.method == "True")]

    fig = plt.figure(figsize=(12, 6))
    g = sns.lineplot(
        data=df_all, 
        x="time", 
        y="particles", 
        hue="method", 
        style="method", 
        alpha=1,
        ci=None
      )
    plt.scatter(
      time_step[target_dist.loc.cpu().numpy()],
      target_dist.obs.cpu().numpy(),
      color="red",
      s=10
    )
    ci_colors = ["blue", "green"]
    plot_methods = [n for n in df_all.method.unique() if n != "True"]
    for method_name, col in zip(plot_methods, ci_colors):
      sub_df_all = df_all.loc[df_all.method==method_name]
      sub_df_all = sub_df_all.sort_values(["method", "time"]).reset_index(drop=True)
      plt.fill_between(data=sub_df_all,
        x="time", y1="lower", y2="upper", alpha=0.2, color=col)
    plt.xlabel("Time Steps", fontsize=25)
    plt.xticks(fontsize=20)
    plt.ylabel("Solution", fontsize=25)
    plt.yticks(fontsize=20)
    plt.legend(fontsize=20, markerscale=2, bbox_to_anchor=(1, 0.4), loc='center left')
    fig.tight_layout()

    fig.savefig(f"{resdir}/solution_all.png")
    fig.savefig(f"{resdir}/solution_all.pdf")
      
    print(f"Saved to {resdir}/solution.pdf")


    ## plot samples
    fig = plt.figure(figsize=(subplot_c*5, subplot_r*5))
    df_list_all = []
    for i, (res, method_name) in enumerate(zip(method_ls, method_names)):
      df_list = []

      particles = res["particles"][-1].cpu().detach().numpy()

      true_samples = res["x_true"].cpu().detach().numpy().reshape((-1,))
      time_step = target_dist.t.cpu().detach().numpy()
      
      dim = particles.shape[1]
      nparticles = particles.shape[0]
      df_new = pd.DataFrame({
        "time": np.tile(time_step, nparticles),
        "particles": particles.reshape((-1,)),
        "sample_ind": np.repeat(range(dim), nparticles),
        "method": method_name,
        "seed": seed
      })
      # append true samples
      df_true = pd.DataFrame({
        "time": time_step,
        "particles": true_samples,
        "sample_ind": 0,
        "method": "True",
        "seed": seed
      })

      df_new = df_new.append(df_true)

      df_list.append(df_new)
      df_list_all.append(df_new)

      df = pd.concat(df_list)
      
      # 90%-CI
      for ind, subdf in enumerate(df_list):
        subdf["lower"] = np.nan
        subdf["upper"] = np.nan
        for t in time_step:
          quantiles = subdf[subdf.time == t].particles.quantile([0.05, 0.95]).to_list()
          subdf.loc[subdf.time == t, "lower"] = quantiles[0]
          subdf.loc[subdf.time == t, "upper"] = quantiles[1]
        
        df_list[ind] = subdf
      
      df = pd.concat(df_list)
      df = df.sort_values(["method", "time"]).reset_index(drop=True)

      plt.subplot(subplot_r, subplot_c, i+1)
      g = sns.lineplot(
        data=df, 
        x="time", 
        y="particles", 
        hue="method", 
        style="method", 
        alpha=1,
        ci=None
      )
      plt.fill_between(data=df.loc[df.method==method_name], x="time", y1="lower", y2="upper", alpha=0.2)
      plt.scatter(
        time_step,
        true_samples,
        color="red",
        s=2*len(method_names)
      )
      plt.title(method_name, fontsize=20)
      plt.xlabel("Time Steps", fontsize=25)
      plt.xticks(fontsize=20)
      plt.ylabel("Samples", fontsize=25)
      plt.yticks(fontsize=20)
      plt.xlim(0, 1)
      plt.ylim(-0.9, 0.1)
      fig.tight_layout()

    fig.savefig(f"{resdir}/particles.png")
    fig.savefig(f"{resdir}/particles.pdf")


    df_all = pd.concat(df_list_all)
    df_all = df_all.loc[(df_all.method == "HMC") | (df_all.method == gsvgd_show)]

    fig = plt.figure(figsize=(12, 6))
    g = sns.lineplot(
      data=df_all, 
      x="time", 
      y="particles", 
      hue="method", 
      style="method", 
      alpha=1,
      ci=None
    )
    plt.scatter(
      time_step,
      true_samples,
      color="red",
      s=2*len(method_names)
    )
    for method_name, col in zip(df_all.method.unique(), ci_colors):
      sub_df_all = df_all.loc[df_all.method==method_name]
      sub_df_all = sub_df_all.sort_values(["method", "time"]).reset_index(drop=True)
      plt.fill_between(data=sub_df_all,
        x="time", y1="lower", y2="upper", alpha=0.2, color=col)
    plt.xlabel("Time Steps", fontsize=25)
    plt.xticks(fontsize=20)
    plt.ylabel("Samples", fontsize=25)
    plt.yticks(fontsize=20)
    plt.legend(fontsize=20, markerscale=2, bbox_to_anchor=(1, 0.4), loc='center left')
    fig.tight_layout()

    fig.savefig(f"{resdir}/particles_all.png")
    fig.savefig(f"{resdir}/particles_all.pdf")

      
  
