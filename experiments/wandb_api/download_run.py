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
import time

parser = argparse.ArgumentParser(add_help=True,
    description='Visualize run output.')
parser.add_argument('--k', help="Key of the run on wandb")
parser.add_argument('--n', type=str, help="Name of the run")
parser.add_argument('--eigen', type=bool, action=argparse.BooleanOptionalAction, default=False)
parser.add_argument('--stats', type=bool, action=argparse.BooleanOptionalAction, default=False)

args, unk = parser.parse_known_args()
args = vars(args)
# Check for unknown options
if unk:
    warn("Unknown arguments:" + str(unk) + ".")

run_name = args["k"]

root_dir = infer_root_dir()
eigen_dir = os.path.join(root_dir, "wandb_run_results")

eigen_name = "eigen.pt"
api = wandb.Api()
run = api.run("philippvonbachmann/VariationalLaplace/" + run_name)
if args["eigen"]:
    for file in run.files():
        if file.name == eigen_name:
            eigen_file = file
            print("Found file")
            break
    eigen_file.download(os.path.join(eigen_dir, run_name))

if args["stats"]:
    dataset = "CIFAR10"

    metrics = ["nll", "ece", "conf"]
    columns = ["_".join([dataset, "laplace", metric]) for metric in metrics]
    save_path = os.path.join(eigen_dir, run_name, "evaluation.csv")
    for i in range(10):
        try:
            df = run.history().iloc[-1][columns]
            os.makedirs(os.path.join(eigen_dir, run_name), exist_ok=True)
            with open(save_path, mode="w") as f:
                f.write(f"Name,{args['n']}\n")
            df.to_csv(save_path, mode="a")
            break
        except KeyError:
            time.sleep(0.5)
            print("+")

            