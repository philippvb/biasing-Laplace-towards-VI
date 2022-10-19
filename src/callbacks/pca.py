import torch
import numpy as np
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader
from torch.nn.utils import vector_to_parameters
import torch.nn.functional as F
from torch import nn
from copy import deepcopy
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.colors import ListedColormap
from tqdm import tqdm
import os
from src.models import compute_diag_ggn


c1 = (38/255, 81/255, 206/255)
c2 = (232/255, 220/255, 87/255)
c3 = (238/255, 238/255, 238/255)
c4 = (0, 0, 0)

def fit_pca(trajectories, n_components=2):
    pca = PCA(n_components=n_components)
    pca.fit(trajectories)
    return pca

@torch.no_grad()
def get_mse(model:nn.Module, dataloader:DataLoader):
    total_loss = 0
    for x, y in dataloader:
        total_loss += F.mse_loss(model(x), y, reduction="sum")
    total_loss /= len(dataloader.dataset)
    return total_loss.log().item()

def get_hessian(model:nn.Module, dataloader:DataLoader):
    # make sure all gradients are set to 
    optimizer = torch.optim.SGD(model.parameters(), lr=1)
    optimizer.zero_grad()
    return compute_diag_ggn(model, nn.MSELoss(), dataloader, print_status=False).mean().log().item()

def plot_loss_landscape(model:nn.Module, dataloader:DataLoader, axs:Axes, fig, pca:PCA, range_x=(-50, 50), range_y=(-30, 30), log=True):
    callback_function = get_mse
    return plot_landscape(model=model, dataloader=dataloader, axs=axs, fig=fig, pca=pca, callback_function=callback_function, range_x=range_x, range_y=range_y)

def plot_hessian_landscape(model:nn.Module, dataloader:DataLoader, axs:Axes, fig, pca:PCA, range_x=(-50, 50), range_y=(-30, 30), log=True):
    callback_function = get_hessian
    return plot_landscape(model=model, dataloader=dataloader, axs=axs, pca=pca, fig=fig, callback_function=callback_function, range_x=range_x, range_y=range_y)

def plot_landscape(model:nn.Module, dataloader:DataLoader, fig:Figure, axs:Axes, pca:PCA, callback_function, range_x=(-50, 50), range_y=(-30, 30)):
    
    # In PCA space
    grid_size = 100
    range_x = np.linspace(*range_x, grid_size)
    range_y = np.linspace(*range_y, grid_size)
    w1s, w2s = np.meshgrid(range_x, range_y)
    thetas_latent = np.stack([w1s.ravel(), w2s.ravel()]).T
    
    # Project back to the parameter space
    thetas = pca.inverse_transform(thetas_latent)
    thetas = torch.from_numpy(thetas).float()
    
    losses = []
    
    _model = deepcopy(model)
    print("Plotting loss landscape")
    for theta in tqdm(thetas):
        vector_to_parameters(theta, _model.parameters())
        losses.append(callback_function(_model, dataloader))
    losses = np.array(losses)
    
    # Marquis de Laplace's colormap
    N = 25
    vals = np.ones((N, 4))
    vals[:, 0] = np.linspace(c1[0], c3[0], N)
    vals[:, 1] = np.linspace(c1[1], c3[1], N)
    vals[:, 2] = np.linspace(c1[2], c3[2], N)
    cmap = ListedColormap(vals)
    cax1 = axs.contourf(
        w1s, w2s, losses.reshape(grid_size, grid_size), levels=20, 
        cmap=cmap, alpha=0.8
    )
    axs.contour(
        w1s, w2s, losses.reshape(grid_size, grid_size), levels=20, 
        alpha=0.5, colors='k'
    )
    axs.set_xticks([])
    axs.set_yticks([])
    



class PCACallback():
    def __init__(self) -> None:
        pass
        self.pca = None

    def fit(self, trajectories):
        self.pca = fit_pca(trajectories=trajectories, n_components=2)

    def _check_pca(self):
        if not self.pca:
            raise RuntimeError("Please fit PCA before calling other functions!")


    def plot_landscape(self, model:nn.Module, dataloader:DataLoader, range_x=(-50, 50), range_y=(-30, 30)):
        self._check_pca()
        self.fig, self.axs = plt.subplots(2, figsize=(10, 20))
        plot_loss_landscape(model=model, dataloader=dataloader, axs=self.axs[0], fig=self.fig, pca=self.pca, range_x=range_x, range_y=range_y)
        self.axs[0].set_title("Loss")
        plot_hessian_landscape(model=model, dataloader=dataloader, axs=self.axs[1], fig=self.fig, pca=self.pca, range_x=range_x, range_y=range_y)
        self.axs[1].set_title("Hessian")
        return self.fig, self.axs

    def save(self, path):
        self._check_pca()
        self.fig.savefig(os.path.join(path, "Loss_landscape_PCA.jpg"))

    def add_trajectory(self, trajectories, label:str):
        self._check_pca()
        trajectories_pca = self.pca.transform(trajectories)
        for ax in self.axs.flat:
            ax.scatter(trajectories_pca[:, 0], trajectories_pca[:, 1], edgecolors=c4, lw=0, alpha=1, zorder=9)
            ax.scatter(trajectories_pca[-1:, 0], trajectories_pca[-1:, 1], marker='*', edgecolors=c4, s=200, lw=1.5, zorder=10, label=label)
            ax.legend()
        