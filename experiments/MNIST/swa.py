import sys

sys.path.append("/Users/philippvonbachmann/Documents/University/WiSe2122/ResearchProject/paper")
sys.path.append("/home/hennig/pvbachmann87/VariationalLaplace")

import pytorch_lightning as pl
import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.seed import seed_everything

from src.datasets import get_dset, TorchVisionDataSet
from src.optimizers import SWACallback
from src.models import ClassificationModel
from src.models.models import SmallCNN
from src.utils.argparser import default_parse, save_args, add_trainer_args, add_checkpoint_args
from src.utils import create_checkpoint_name
from src.callbacks import setup_callbacks



args = default_parse([add_trainer_args, add_checkpoint_args, TorchVisionDataSet.add_model_specific_args, ClassificationModel.add_model_specific_args, SmallCNN.add_model_specific_args, SWACallback.add_model_specific_args])
args_trainer = args
args = vars(args)
seed_everything(args["seed"])
root_dir = create_checkpoint_name(args["run_name"])

dataset = get_dset(**args)
args["image_dims"] = dataset.get_image_dims()

neural_network = SmallCNN(**args)
model = ClassificationModel(model=neural_network, **args)

wandb_logger = WandbLogger(name=args["run_name"], project='VariationalLaplace')
callback = setup_callbacks(root_dir, callbacks=["parameters"], **args)
callback.append(SWACallback(**args))
trainer = pl.Trainer.from_argparse_args(args_trainer, callbacks=callback, logger=wandb_logger)
trainer.fit(model, datamodule=dataset)
wandb.config.update(args)
save_args(root_dir, args)