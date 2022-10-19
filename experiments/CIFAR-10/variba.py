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
from src.models.models import VariationalWideResNet
from src.utils.argparser import default_parse, add_trainer_args, save_args, add_checkpoint_args
from src.utils import create_checkpoint_name
from src.callbacks import setup_callbacks

args = default_parse([add_trainer_args, add_checkpoint_args , TorchVisionDataSet.add_model_specific_args, ClassificationDiagVB.add_model_specific_args, VariationalWideResNet.add_model_specific_args])
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
args["image_dims"] = dataset.get_image_dims()
args["lr_decay_cosine_steps"] = len(dataset.train_dataloader()) * args["max_epochs"]
args["dset_size"] = len(dataset.train_dataloader().dataset)
args["scheduler_step_per_batch"] = True
network_normal  = VariationalWideResNet(**args)
model = ClassificationDiagVB(model=network_normal, model_normal=None, **args)

callbacks = setup_callbacks(root_dir, callbacks=["statistics"], **args, variational=True, test_n_samples=100)
trainer = pl.Trainer.from_argparse_args(args_trainer, callbacks=callbacks, logger=wandb_logger)
trainer.fit(model=model, datamodule=dataset)
trainer.test(model, datamodule=dataset)

if not args["no_log"]:
    wandb.config.update(args, allow_val_change=True)
save_args(root_dir, args)