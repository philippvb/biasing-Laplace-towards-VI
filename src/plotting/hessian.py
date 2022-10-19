from matplotlib import pyplot as plt
import torch
import wandb
import numpy as np

def plot_eigen(eigen:torch.Tensor, ax:plt.Axes, bins=None):
    eigen = eigen.flatten().cpu().numpy()
    ax.hist(eigen, bins=bins)
    ax.set_xlabel("Eigenvalue")
    ax.set_ylabel("Count")
    return ax