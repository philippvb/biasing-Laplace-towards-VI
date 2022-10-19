import os
import sys

sys.path.append("/Users/philippvonbachmann/Documents/University/WiSe2122/ResearchProject/paper")
sys.path.append("/home/hennig/pvbachmann87/VariationalLaplace")

import pytorch_lightning as pl
import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.seed import seed_everything
from copy import deepcopy

from src.datasets import ToyDataSet2D
from src.models import NeuralNetwork, VariationalNetwork
from src.models.models import ToyModel2D, VariationalToyModel2D
from src.utils.argparser import default_parse, add_trainer_args, save_args, add_checkpoint_args
from src.utils import create_checkpoint_name
from src.callbacks import setup_callbacks, ToyVisualization
from src.datasets import get_dset
import torch
from src.callbacks import PCACallback
from src.optimizers import SWACallback
from src.models.utils import copy_params_to_bnn

args = default_parse([add_trainer_args, add_checkpoint_args, ToyDataSet2D.add_model_specific_args, NeuralNetwork.add_model_specific_args])
args_trainer = args
args = vars(args)
seed_everything(args["seed"])
root_dir = create_checkpoint_name(args["run_name"])
if not args["no_log"]:
    wandb_logger = WandbLogger(name=args["run_name"], project='VariationalLaplace')
else:
    wandb_logger = None

dataset = get_dset(**args)
dataset.setup("fit")
# args["lr_decay_cosine_steps"] = len(dataset.train_dataloader())

args["scheduler_name"] = "step"
args["lr_decay_step"] = 500
args["lr_decay_factor"] = 0.1
neural_network = ToyModel2D(**args)

model_map = NeuralNetwork(model=deepcopy(neural_network), loss_fun=torch.nn.MSELoss() ,**args)
callbacks_map = setup_callbacks(root_dir, callbacks=["trajectory"], **args)
callbacks_map.append(ToyVisualization(""))
trainer_map = pl.Trainer.from_argparse_args(args_trainer, callbacks=callbacks_map, logger=wandb_logger)
trainer_map.fit(model=model_map, datamodule=dataset)
trainer_map.test(datamodule=dataset)



# args["tau"] = 1
# variational_network = VariationalToyModel2D(**args)
# # variational_network_normal = deepcopy(model_map.model)
# variational_network_normal = deepcopy(neural_network)
# # args_trainer.max_epochs = 3000
# copy_params_to_bnn(variational_network, variational_network_normal)
# model_vb = VariationalNetwork(model=variational_network, model_normal=variational_network_normal, loss_fun=torch.nn.MSELoss(), dset_size=len(dataset.train_dataloader().dataset),**args)
# callbacks_vb = setup_callbacks(root_dir, callbacks=["parameters", "trajectory"], variational=True, **args)
# trainer_vb = pl.Trainer.from_argparse_args(args_trainer, callbacks=callbacks_vb, logger=wandb_logger)
# trainer_vb.fit(model=model_vb, datamodule=dataset)


# args_trainer.max_epochs = 500
# model_swa = NeuralNetwork(model=deepcopy(model_map.model), loss_fun=torch.nn.MSELoss() ,**args)
# callbacks_swa = setup_callbacks(root_dir, callbacks=["parameters", "trajectory"], **args)
# callbacks_swa.append(SWACallback(swa_epoch_start=0., swa_lrs=1e-2, annealing_epochs=100))
# trainer_swa = pl.Trainer.from_argparse_args(args_trainer, callbacks=callbacks_swa, logger=wandb_logger)
# trainer_swa.fit(model=model_swa, datamodule=dataset)

# # model_map_2 = NeuralNetwork(model=deepcopy(model_map.model), loss_fun=torch.nn.MSELoss() ,**args)
# # callbacks_map_2 = setup_callbacks(root_dir, callbacks=["parameters", "trajectory"], **args)
# # trainer_map_2 = pl.Trainer.from_argparse_args(args_trainer, callbacks=callbacks_map_2, logger=wandb_logger)
# # trainer_map_2.fit(model=model_map_2, datamodule=dataset)


# pca = PCACallback()
# pca.fit(callbacks_map[2].get_trajectory())
# xy_range = (-50, 50)
# pca.plot_landscape(model=model_map.model, dataloader=dataset.train_dataloader(), range_x=xy_range, range_y=xy_range)
# pca.add_trajectory(callbacks_map[2].get_trajectory(), label="MAP")
# pca.add_trajectory(callbacks_swa[2].get_trajectory(), label="SWA")
# pca.add_trajectory(callbacks_vb[2].get_trajectory(), label="VB")
# # pca.add_trajectory(callbacks_map_2[2].get_trajectory(), label="MAPLONG")
# pca.save("")


# if not args["no_log"]:
#     wandb.config.update(args)
# save_args(root_dir, args, use_wandb=not args["no_log"])
