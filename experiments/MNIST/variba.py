import os
import sys

import wandb
sys.path.append("/Users/philippvonbachmann/Documents/University/WiSe2122/ResearchProject/paper")
sys.path.append("/home/hennig/pvbachmann87/VariationalLaplace")

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.seed import seed_everything
import wandb

from src.datasets import MNISTDataloader
from src.models import ClassificationDiagVB
from src.models.models import SmallCNN, VariationalSmallCNN
from src.utils.argparser import add_checkpoint_args, add_trainer_args, default_parse
from src.utils import create_checkpoint_name
from src.callbacks import setup_callbacks
from src.datasets import get_dset

args = default_parse([add_trainer_args,add_checkpoint_args, MNISTDataloader.add_model_specific_args, ClassificationDiagVB.add_model_specific_args, SmallCNN.add_model_specific_args])
args_trainer = args
args = vars(args)
args["variational"] = True
seed_everything(args["seed"])
root_dir = create_checkpoint_name(args["run_name"])
wandb_logger = WandbLogger(name=args["run_name"], project='VariationalLaplace')

dataset = get_dset(**args)
args["image_dims"] = dataset.get_image_dims()
dataset.setup("fit")
args["dset_size"] = len(dataset.train_dataloader().dataset)
network = VariationalSmallCNN(**args)
network_normal = SmallCNN(**args)
model = ClassificationDiagVB(model=network, model_normal=network_normal, **args)

callbacks = setup_callbacks(root_dir, callbacks=["parameters"], variational=True, **args)
trainer = pl.Trainer.from_argparse_args(args_trainer, callbacks=callbacks, logger=wandb_logger)
trainer.fit(model=model, datamodule=dataset)