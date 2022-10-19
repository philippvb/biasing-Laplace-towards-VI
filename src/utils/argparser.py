import argparse
from glob import glob
from typing import Tuple
from warnings import warn    
import json
import os
from numpy import arange
import wandb

def default_parse(add_to_parser=[]):
    parser = argparse.ArgumentParser(add_help=True,
    description='Train model.')
    for fun in add_to_parser:
        parser = fun(parser)
    args, unk = parser.parse_known_args()
    if unk:
        warn("Unknown arguments:" + str(unk) + ".")
    return args

def add_trainer_args(parent_parser):
    parser = parent_parser.add_argument_group("Trainer")
    parser.add_argument('--run_name', type=str)
    parser.add_argument('--no_log', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument("--max_epochs", type=int)
    parser.add_argument("--gpus", type=int, default=1)
    return parent_parser

def add_checkpoint_args(parent_parser):
    parser = parent_parser.add_argument_group("Checkpoint")
    parser.add_argument('--every_n_epochs', type=int, default=None)
    return parent_parser

def save_args(path, args, use_wandb=True):
    if use_wandb:
        args["id"] = wandb.run.id
    if not os.path.exists(path):
        os.makedirs(path)
    with open(os.path.join(path, 'args.json'), 'w') as f:
        json.dump(args, f, indent='\t')

def checkpoint_parse():
    def add_run_name(parser):
        parser.add_argument('--load_dir', type=str)
        return parser
    args = vars(default_parse([add_run_name]))
    path = args["load_dir"]
    with open(os.path.join(path, "args.json")) as f:
        args = json.load(f)
    args["load_dir"] = path
    return args

def load_checkpoint(add_to_parser=[]):
    parser = argparse.ArgumentParser(add_help=True,
    description='Load model.')
    parser.add_argument("--load_dir", type=str)
    for fun in add_to_parser:
        parser = fun(parser)
    args, unk = parser.parse_known_args()
    args = vars(args)
    if unk:
        warn("Unknown arguments:" + str(unk) + ".")
    with open(os.path.join(args["load_dir"], "args.json")) as f:
        new_args = json.load(f)

    for name in glob(args["load_dir"] + "*.ckpt"):
        print("Found checkpiont", name)
        args["checkpoint_name"] = name
        break
    return {**args, **new_args}

def parse_args(m=None, d=None, t=None, o=None, add_to_parser=None) -> Tuple[dict, dict, dict, dict]:
    parser = argparse.ArgumentParser(add_help=True,
        description='Train model to predict rage from the raw ecg tracing.')
    parser.add_argument('--m', help="Path to the json config file of the model")
    parser.add_argument('--n', help="Run name", default="Test")
    parser.add_argument('--d', help="Path to the json config file of the dataset")
    parser.add_argument('--t', help="Path to the json config file of the train run")
    parser.add_argument('--o', help="Path to the json config file of the other")
    if add_to_parser:
        add_to_parser(parser)

    args, unk = parser.parse_known_args()
    args = vars(args)
    print(unk)
    # Check for unknown options
    if unk:
        warn("Unknown arguments:" + str(unk) + ".")

    # load the model dict
    model_json = m if m else args["m"]
    with open(model_json, "r") as f:
        model_args = json.load(f)
    
    # load the dataset dict
    dset_json = d if d else args["d"]
    with open(dset_json, "r") as f:
        ds_args = json.load(f)

    train_json = t if t else args["t"]
    with open(train_json, "r") as f:
        train_args = json.load(f)
    
    other_json = o if o else args["o"]
    with open(other_json, "r") as f:
        other_args = json.load(f)
    other_args["run_name"] = args["n"]

    del args["d"]
    del args["m"]
    del args["n"]
    del args["t"]
    del args["o"]
    other_args = {**other_args, **args}
        
    return train_args, model_args, ds_args, other_args


# def save_args(train_args, model_args, ds_args):
#     try:
#         folder = train_args["folder"] + train_args["output_foldername"]
#     except:
#         print("The json for the train config must contain the items \"folder\" and \"output_foldername\"")

#     with open(os.path.join(folder, 'model.json'), 'w') as f:
#         json.dump(model_args, f, indent='\t')
#     with open(os.path.join(folder, 'train_config.json'), 'w') as f:
#         json.dump(train_args, f, indent='\t')
#     with open(os.path.join(folder, 'dataset.json'), 'w') as f:
#         json.dump(ds_args, f, indent='\t')
