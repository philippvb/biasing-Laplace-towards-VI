import sys, os
sys.path.append("/Users/philippvonbachmann/Documents/University/WiSe2122/ResearchProject/paper")
sys.path.append("/home/hennig/pvbachmann87/VariationalLaplace")
import wandb
import torch
from src.plotting.hessian import plot_eigen
from matplotlib import pyplot as plt
from src.utils import infer_root_dir
import argparse
from warnings import warn

parser = argparse.ArgumentParser(add_help=True,
    description='Visualize run output.')
parser.add_argument('--k', help="Key of the run on wandb")
parser.add_argument('--n', type=str, help="Name of the run")
parser.add_argument('--d', help="Wether to download the run", action=argparse.BooleanOptionalAction)

args, unk = parser.parse_known_args()
args = vars(args)
# Check for unknown options
if unk:
    warn("Unknown arguments:" + str(unk) + ".")

run_name = args["k"]
download = args["d"]

root_dir = infer_root_dir()
eigen_dir = os.path.join(root_dir, "wandb_run_results")
eigen_name = "eigen.pt"

if download:
    api = wandb.Api()
    run = api.run("philippvonbachmann/VariationalLaplace/" + run_name)

    for file in run.files():
        if file.name == eigen_name:
            eigen_file = file
            print("Found file")
            break
    eigen_file.download(os.path.join(eigen_dir, run_name))
eigenvalues = torch.load(os.path.join(eigen_dir, run_name, eigen_name))

print("Shape", eigenvalues.shape)

fig, axs = plt.subplots(1)
axs.set_title(args["n"])
axs.set_xlabel("Eigenvalue")
axs.set_ylabel("Count")
lower, upper = torch.tensor(1e-10).log10(), torch.tensor(0.2).log10()
# print(eigenvalues.min(), eigenvalues.max())
# lower, upper = eigenvalues.min().log10(), eigenvalues.max().log10()
plot_eigen(eigenvalues, axs, bins=torch.logspace(lower, upper, 100))
# lower, upper = 0, 0.00002
# plot_eigen(eigenvalues, axs, bins=torch.linspace(lower, upper, 100))
# axs.set_ylim(0, 400000)
plt.savefig(os.path.join(eigen_dir, run_name, "Eigen_" + args["n"] +  ".jpg"))
