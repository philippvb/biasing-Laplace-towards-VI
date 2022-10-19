import wandb
import torch
import os
import numpy as np

def save_tensor_to_wandb(tensor:torch.Tensor, key:str):
    key = key.lower() + ".pt"
    save_path = os.path.join(wandb.run.dir, key)
    torch.save(tensor, save_path)


def log_eigen(eigen:torch.Tensor, bins, key="ExactHessian"):
    bins = bins.numpy()
    classes = np.digitize(eigen, bins)
    bin_id = np.arange(len(bins))
    eigen_count = np.array([np.sum(classes == i) for i in bin_id]).astype("int")
    table = wandb.Table(columns=["Eigenvalues","Count"], data=np.array([bins, eigen_count]).T)
    wandb.log({key: wandb.plot.line(table, "Eigenvalues", "Count")})