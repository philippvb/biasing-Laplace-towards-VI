from torch.optim import lr_scheduler
from src.optimizers import EntropySGD
from src.models.basic import NeuralNetwork
from torch import nn
import argparse

class EntropyNeuralNetwork(NeuralNetwork):

    def __init__(self, momentum=0, damp=0, weight_decay=0, nesterov=True, L=10, noise=1e-4, gamma=1e-2, scoping=0, lr=0.01, *args, **kwargs) -> None:
        """_summary_

        Args:
            momentum (int, optional): _description_. Defaults to 0.
            damp (int, optional): _description_. Defaults to 0.
            weight_decay (int, optional): _description_. Defaults to 0.
            nesterov (bool, optional): _description_. Defaults to True.
            L (int, optional): How many updates within the Langevin dynamics loop. Defaults to 20 as suggested by authors.
            eps (_type_, optional): _description_. Defaults to 1e-4.
            gamma (_type_, optional): Controls how much strength is layed on the width. Defaults to 1e-2.
            scoping (int, optional): Wether gamma is increasd over the course of training. Defaults to 0.
        """
        super().__init__(*args, **kwargs)
        # for CIFAR-10, they authors use:
        # weight_decay = 1e-3
        # L = 20
        # gamma = 0.03 or 1e-4
        # scoping = 1e-3
        # noise = 1e-4



        self.momentum = momentum 
        self.damp=damp 
        self.weight_decay=weight_decay
        self.nesterov=nesterov
        self.L=L
        self.noise=noise
        self.gamma=gamma
        self.scoping=scoping # formular for scoping is gamma (1+scoping ^t)
        self.lr = lr

        self.automatic_optimization = False

    def get_hparams(self):
        return super().get_hparams() + ["damp", "L", "noise", "gamma", "scoping"]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("EntropyNetwork")

        def str_to_float_or_int(str):
            str = float(str)
            if (str % 1) == 0:
                str = int(str)
            return str

        parser.add_argument("--damp", type=float, default=0)
        parser.add_argument("--L", type=int, default=10)
        parser.add_argument("--noise", type=float, default=1e-4)
        parser.add_argument("--gamma", type=float, default=1e-2)
        parser.add_argument("--scoping", type=float, default=0)
        parent_parser = NeuralNetwork.add_model_specific_args(parent_parser)
        return parent_parser

    def configure_optimizers(self):
        optimizer = EntropySGD(self.parameters(), 
        config={"momentum":self.momentum, "damp":self.damp, "weight_decay":self.weight_decay, "lr": self.lr,
        "nesterov": self.nesterov, "L": self.L, "eps": self.noise, "g0": self.gamma, "g1": self.scoping})

        if self.scheduler_class:
            scheduler = self.scheduler_class(optimizer)
            scheduler_config = {
                "scheduler": scheduler,
                "interval": "step" if self.scheduler_step_per_batch else "epoch"
            }
            print("Using scheduler")
            return [optimizer], [scheduler_config]
        return optimizer
        


    def training_step(self, batch):
        self.optimizers().zero_grad()
        def closure():
            data, target = batch
            prediction = self.model(data)
            loss = self.loss_fun(prediction, target)
            loss.backward()
            return loss, None

        loss, _ = self.optimizers().optimizer.step(closure=closure, model=self.model, criterion=nn.CrossEntropyLoss())
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("learning_rate", self.optimizers().param_groups[0]["lr"], on_step=True, on_epoch=True, prog_bar=False)
        return loss

    def on_epoch_end(self) -> None:
        schedulers = self.lr_schedulers()
        schedulers.step(self.current_epoch + 1)
        return super().on_epoch_end()



        

        
        
