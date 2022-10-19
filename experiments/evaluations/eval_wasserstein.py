import os
import sys
import argparse
from warnings import warn
import torch
sys.path.append("/Users/philippvonbachmann/Documents/University/WiSe2122/ResearchProject/paper")
sys.path.append("/home/hennig/pvbachmann87/VariationalLaplace")

from pytorch_lightning.plugins.io import TorchCheckpointIO
from matplotlib import pyplot as plt

from src.utils import infer_root_dir
from src.metrics.distance import wasserstein_diag_gauss

parser = argparse.ArgumentParser(add_help=True,
    description='Evaluate Wasserstein metric')
parser.add_argument('--n', help="Names of the runs", nargs="+", type=str)

args, unk = parser.parse_known_args()
args = vars(args)
# Check for unknown options
if unk:
    warn("Unknown arguments:" + str(unk) + ".")

root_dir = infer_root_dir()

checkpoint_loader = TorchCheckpointIO()
parameters = {}
for run_name in args["n"]:
    parameters[run_name] = checkpoint_loader.load_checkpoint(os.path.join(root_dir, run_name, "parameters.pt"))

wasser_dist = {}
n = len(args["n"])
for i in range(n):
    for j in range(i+1, n):
        run1 = args["n"][i]
        run2 = args["n"][j]
        print(torch.any(torch.isnan(parameters[run1]["mean"])))
        print(torch.any(torch.isnan(parameters[run1]["variance"])))
        print(torch.any(torch.isnan(parameters[run2]["mean"])))
        print(torch.any(torch.isnan(parameters[run2]["variance"])))
        wasser_dist[f"{run1}-{run2}"] = wasserstein_diag_gauss(parameters[run1]["mean"], parameters[run1]["variance"], parameters[run2]["mean"], parameters[run2]["variance"]).item()

print(wasser_dist)
fig, axs = plt.subplots(1)
axs.hist(wasser_dist.keys(), wasser_dist.values())
axs.set_ylabel("Wasserstein distance")
plt.savefig("test.jpg")





