import os
import sys
sys.path.append("/Users/philippvonbachmann/Documents/University/WiSe2122/ResearchProject/paper")
sys.path.append("/home/hennig/pvbachmann87/VariationalLaplace")

import pytorch_lightning as pl
import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.seed import seed_everything

from src.datasets import TorchVisionDataSet, get_dset
from src.models import ClassificationDiagVB
from src.models.models import VariationalWideResNet, WideResNet
from src.utils.argparser import load_checkpoint
from src.utils import create_checkpoint_name
from src.callbacks import setup_callbacks
from src.callbacks.evaluation import VariationalStatisticsCallback

args = load_checkpoint()
seed_everything(args["seed"])
root_dir = create_checkpoint_name(args["run_name"])

dataset = get_dset(**args)
args["image_dims"] = dataset.get_image_dims()
neural_network = VariationalWideResNet(**args)
model = ClassificationDiagVB.load_from_checkpoint(args["checkpoint_name"], model=neural_network, model_normal=None, scheduler_step_per_batch = None)
callbacks = [VariationalStatisticsCallback()]
trainer = pl.Trainer(callbacks=callbacks, gpus=args["gpus"])
trainer.test(model, datamodule=dataset)
