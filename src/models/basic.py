import argparse
from email.policy import default
import pytorch_lightning as pl
from torch import nn, optim
import torch
from torch.utils.data import DataLoader
from torch.optim import Optimizer
import argparse
from torch.distributions import Normal
from torch.nn.utils import parameters_to_vector
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, SequentialLR
from torchcontrib.optim import SWA

class NeuralNetwork(pl.LightningModule):
    def __init__(self, model:nn.Module, loss_fun:nn.Module, optimizer_name:Optimizer="SGD", scheduler_name:str="STEP", prior_std:float=None, **kwargs) -> None:
        super().__init__()
        self.model = model
        self.loss_fun = loss_fun
        self.optimizer_class = self.get_optimizer(optimizer_name, **kwargs)
        self.scheduler_class = self.get_scheduler(scheduler_name, **kwargs) if scheduler_name else None
        self.scheduler_step_per_batch = kwargs["scheduler_step_per_batch"]
        self.annealing_epochs = kwargs["annealing_epochs"]
        self.use_swa = kwargs["use_swa"]
        self.swa_epoch_start = kwargs["swa_epoch_start"]

        self.save_hyperparameters(*self.get_hparams())
        if prior_std:
            self.prior_dist = Normal(0, prior_std)
        else:
            self.prior_dist = None

    def get_prior(self) -> torch.Tensor:
        if self.prior_dist:
            return - self.prior_dist.log_prob(parameters_to_vector(self.model.parameters())).mean()
        else:
            return 0

    def get_hparams(self):
        return ["lr", "optimizer_name", "momentum", "weight_decay", "nesterov",
            "scheduler_name", "scheduler_step_per_batch", "lr_decay_step", "lr_decay_factor", "lr_decay_cosine_steps",
            "use_swa", "annealing_epochs", "swa_epoch_start"]

    def get_optimizer(self, optimizer_name:str, use_swa=False, **kwargs):
        optimizer_name = optimizer_name.upper()
        if optimizer_name == "SGD":
            optimizer_class =  lambda params: optim.SGD(params, lr=kwargs["lr"], momentum=kwargs["momentum"], weight_decay=kwargs["weight_decay"], nesterov=kwargs["nesterov"])
        else:
            raise ValueError(f"Optimizer with name {optimizer_name} not found")
        if use_swa:
            def wrap_swa(params):
                swa = SWA(optimizer_class(params))
                # add missing attributes according to https://github.com/pytorch/contrib/issues/36
                swa.param_groups = swa.optimizer.param_groups
                swa.state = swa.optimizer.state
                swa.defaults=swa.optimizer.defaults
                return swa
            return wrap_swa
        else:
            return optimizer_class

    def get_scheduler(self, scheduler_name:str, use_swa=False, **kwargs):
        scheduler_name = scheduler_name.upper()
        if scheduler_name == "STEP":
            scheduler_class = lambda optimizer: optim.lr_scheduler.StepLR(optimizer, step_size=kwargs["lr_decay_step"], gamma=kwargs["lr_decay_factor"])
        elif scheduler_name == "COSINE":
            scheduler_class = lambda optimizer: optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=kwargs["lr_decay_cosine_steps"], eta_min=kwargs["lr_decay_eta_min"])
        else:
            raise ValueError(f"Scheduler with name {scheduler_name} not found")

        if use_swa:
            # chain with second scheduler
            if type(kwargs["swa_epoch_start"]) == float:
                raise NotImplementedError
            if scheduler_name != "COSINE":
                raise NotImplementedError
            t_0 = kwargs["annealing_steps"] if kwargs["scheduler_step_per_batch"] else kwargs["annealing_epochs"]
            cosine_warm_restart = lambda optimizer: CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=t_0, T_mult=1, eta_min=0)
            return lambda optimizer: SequentialLR(optimizer=optimizer, schedulers=[scheduler_class(optimizer=optimizer), cosine_warm_restart(optimizer=optimizer)], milestones=[kwargs["lr_decay_cosine_steps"]]) # TODO: clean separation step or epoch up properly
        else:
            return scheduler_class
            

    @staticmethod
    def add_model_specific_args(parent_parser):
        optim_parser = parent_parser.add_argument_group("NeuralNetwork-optimizer")
        optim_parser.add_argument("--optimizer_name", type=str, default="SGD")
        optim_parser.add_argument("--lr", type=float, default=1e-3)
        optim_parser.add_argument("--prior_std", type=float, default=None)
        optim_parser.add_argument("--momentum", type=float, default=0)
        optim_parser.add_argument("--weight_decay", type=float, default=0)
        optim_parser.add_argument('--nesterov', action=argparse.BooleanOptionalAction, default=False)
        optim_parser.add_argument('--use_swa', action=argparse.BooleanOptionalAction, default=False)


        scheduler_parser = parent_parser.add_argument_group("NeuralNetwork-scheduler")
        def str_to_float_or_int(str):
            str = float(str)
            if (str % 1) == 0:
                str = int(str)
            return str
        scheduler_parser.add_argument("--scheduler_name", type=str, default=None)
        scheduler_parser.add_argument("--scheduler_step_per_batch", type=bool, action=argparse.BooleanOptionalAction, default=False)
        scheduler_parser.add_argument("--lr_decay_step", type=str_to_float_or_int, default=None)
        scheduler_parser.add_argument("--lr_decay_factor", type=float, default=None)
        scheduler_parser.add_argument("--lr_decay_cosine_steps", type=int, default=None)
        scheduler_parser.add_argument("--lr_decay_eta_min", type=float, default=0)

        scheduler_parser.add_argument("--annealing_epochs", type=int, default=None)
        scheduler_parser.add_argument("--annealing_steps", type=int, default=None)
        scheduler_parser.add_argument("--swa_epoch_start", type=str_to_float_or_int, default=0.75)

        return parent_parser

    def on_train_epoch_end(self) -> None:
        # update swa
        current_epoch = self.trainer.current_epoch + 2 # end of previous epoch when 1 based
        # check that we use swa
        if self.use_swa \
            and (((current_epoch - self.swa_epoch_start) % self.annealing_epochs) == 0) and (current_epoch >= self.swa_epoch_start): # current epoch is checkpoint of swa
            print(f"Adding SWA checkpoint at end of {self.trainer.current_epoch} wit lr of {self.optimizers().param_groups[0]['lr']}")
            self.optimizers().update_swa()
        return super().on_train_epoch_end()

    def on_train_end(self) -> None:
        #swap swa and sgd
        if self.use_swa:
            self.optimizers().swap_swa_sgd()
        return super().on_train_end()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        data, target = batch
        prediction = self.model(data)
        loss = self.loss_fun(prediction, target)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("learning_rate", self.optimizers().param_groups[0]["lr"], on_step=True, on_epoch=True, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        data, target = batch
        prediction = self.model(data)
        loss = self.loss_fun(prediction, target)
        accuracy = (prediction.argmax(dim=-1) == target).float().mean()
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val_acc", accuracy, on_step=False, on_epoch=True, prog_bar=False)
        return loss

    def configure_optimizers(self):
        optimizer = self.optimizer_class(self.model.parameters())
        if self.scheduler_class:
            scheduler = self.scheduler_class(optimizer)
            scheduler_config = {
                "scheduler": scheduler,
                "interval": "step" if self.scheduler_step_per_batch else "epoch"
            }

            return [optimizer], [scheduler_config]
        return optimizer

    def predict_step(self, batch, batch_idx):
        data, target = batch
        prediction = self.model(data)
        return prediction, target

    def test_step(self, batch, batch_idx):
        pass

    @torch.no_grad()
    def predict_from_dataloader(self, dataloader:DataLoader) -> tuple[torch.Tensor, torch.Tensor]:
        prediction_list = []
        target_list = []
        for x, target in dataloader:
            prediction = self.forward(x)
            prediction_list.append(prediction)
            target_list.append(target)
        prediction_list = torch.cat(prediction_list, dim=0)
        target_list = torch.cat(target_list, dim=0)

        return prediction_list, target_list
