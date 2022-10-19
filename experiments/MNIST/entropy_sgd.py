import sys
sys.path.append("/Users/philippvonbachmann/Documents/University/WiSe2122/ResearchProject/paper")
sys.path.append("/home/hennig/pvbachmann87/VariationalLaplace")

import pytorch_lightning as pl
import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.seed import seed_everything

from src.datasets import get_dset, TorchVisionDataSet
from src.models import ClassificationEntropy
from src.models.models import SmallCNN
from src.callbacks import setup_callbacks
from src.utils import create_checkpoint_name
from src.utils.argparser import add_checkpoint_args, default_parse, add_trainer_args, save_args


args = default_parse([add_trainer_args, add_checkpoint_args, TorchVisionDataSet.add_model_specific_args, ClassificationEntropy.add_model_specific_args, SmallCNN.add_model_specific_args])
args_trainer = args
args = vars(args)
seed_everything(args["seed"])
root_dir = create_checkpoint_name(args["run_name"])
wandb_logger = WandbLogger(name=args["run_name"], project='VariationalLaplace')

dataset = get_dset(**args)
args["image_dims"] = dataset.get_image_dims()
neural_network = SmallCNN(**args)
model = ClassificationEntropy(model=neural_network, **args)

callbacks = setup_callbacks(root_dir, callbacks=["parameters"], **args)
trainer = pl.Trainer.from_argparse_args(args_trainer, callbacks=callbacks, logger=wandb_logger)
trainer.fit(model=model, datamodule=dataset)

wandb.config.update(args)
save_args(root_dir, args)
