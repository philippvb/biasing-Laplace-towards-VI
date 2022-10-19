import sys
sys.path.append("/Users/philippvonbachmann/Documents/University/WiSe2122/ResearchProject/paper")
sys.path.append("/home/hennig/pvbachmann87/VariationalLaplace")

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger

from src.datasets import  get_dset
from src.models import ClassificationModel
from src.models.models import SmallCNN
from src.utils.argparser import checkpoint_parse
from src.models.utils import forward_dataloader
from src.metrics.classification import confidence
from src.callbacks.evaluation import EvalLaplace, FitLaplace, StatisticsCallback, evaluate_statistics, fit_laplace_and_predict

args = checkpoint_parse()
dataset = get_dset(**args)
dataset.setup_all()
neural_network = SmallCNN(**args)
model = ClassificationModel.load_from_checkpoint(args["load_dir"] + "epoch=19-step=39.ckpt", model=neural_network)
callback = [FitLaplace(), StatisticsCallback(), EvalLaplace()]
logger = CSVLogger(save_dir=args["load_dir"], name="eval_laplace")
trainer = Trainer(gpus=args["gpus"], callbacks=callback, logger=logger)
print("MAP")
trainer.test(model, datamodule=dataset)
trainer = Trainer(gpus=args["gpus"], callbacks=callback[1:], logger=logger)

print("FashionMNIST")
ood_args = dict(args)
ood_args["dset"] = "fashionmnist"
fashion_mnist = get_dset(**ood_args)
fashion_mnist.setup_all()
trainer.test(model, datamodule=fashion_mnist)

print("Noise")
noise_args = dict(args)
noise_args["dset"] = "mnistnoise"
noise_dataset = get_dset(**noise_args)
noise_dataset.setup_all()
trainer.test(model, datamodule=noise_dataset)



