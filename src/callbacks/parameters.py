import os
from typing import Tuple, List

import torch
from torch.nn.utils import parameters_to_vector
from laplace import Laplace
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.plugins.io import TorchCheckpointIO

from src.models.basic import NeuralNetwork
from src.models.variational import VariationalNetwork
from src.models.hessian import compute_diag_ggn, compute_diag_hessian
from src.models.utils import get_blitz_params
from src.utils.wandb import log_eigen, save_tensor_to_wandb


class ParameterLogger(Callback):
    def __init__(self, path:str, to_wandb=True) -> None:
        super().__init__()
        self.path = path
        self.to_wandb = to_wandb
        self.checkpoint = TorchCheckpointIO()

    def on_train_end(self, trainer: pl.Trainer, pl_module:pl.LightningModule) -> None:
        pl_module.eval()
        mean, variance, eigen = self.get_parameters(trainer, pl_module)
        mean, variance, eigen = mean.cpu(), variance.cpu(), eigen.cpu()
        self.checkpoint.save_checkpoint({"mean": mean, "variance": variance, "hessian": eigen}, os.path.join(self.path, "parameters.pt"))
        if self.to_wandb:
            log_eigen(eigen, bins=torch.logspace(eigen.min().log10(), eigen.max().log10(), 100))
            save_tensor_to_wandb(eigen, "eigen")
        pl_module.train()
        return super().on_train_end(trainer, pl_module)

    def get_parameters(self, trainer: pl.Trainer, pl_module:pl.LightningModule) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.get_mean(trainer, pl_module), self.get_var(trainer, pl_module), self.get_eigen(trainer, pl_module)

    def get_mean(self, trainer: pl.Trainer, pl_module:NeuralNetwork) -> torch.Tensor:
        raise NotImplementedError

    def get_var(self, trainer: pl.Trainer, pl_module:NeuralNetwork) -> torch.Tensor:
        raise NotImplementedError

    def get_eigen(self, trainer: pl.Trainer, pl_module:NeuralNetwork) -> torch.Tensor:
        raise NotImplementedError



class StandardParameterLogger(ParameterLogger):
    def __init__(self, path:str, method:str="hessian", to_wandb=True) -> None:
        super().__init__(path, to_wandb=to_wandb)
        self.method = method

    def get_mean(self, trainer: pl.Trainer, pl_module:NeuralNetwork) -> torch.Tensor:
        return parameters_to_vector(pl_module.model.parameters())

    def get_var(self, trainer: pl.Trainer, pl_module: NeuralNetwork) -> torch.Tensor:
        train_dataloader = trainer.train_dataloader
        laplace_model = Laplace(pl_module.model, likelihood="classification", hessian_structure="diag")
        laplace_model.fit(train_dataloader)
        return laplace_model.posterior_variance
        
    def get_eigen(self, trainer: pl.Trainer, pl_module: NeuralNetwork) -> torch.Tensor:
        train_dataloader = trainer.train_dataloader
        if self.method.upper() =="HESSIAN":
            return compute_diag_hessian(pl_module.model, pl_module.loss_fun, train_dataloader)
        elif self.method.upper() =="GGN":
            return compute_diag_ggn(pl_module.model, pl_module.loss_fun, train_dataloader)
        else:
            raise ValueError(f"The desired method {self.method} method can't be used for Eigenvalue computation!")
        

    def get_parameters(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        mean, eigen = self.get_mean(trainer, pl_module), self.get_eigen(trainer, pl_module)
        return mean, 1/eigen, eigen




class VariationalParameterLogger(ParameterLogger):
    def __init__(self, path: str, to_wandb=True) -> None:
        super().__init__(path, to_wandb=to_wandb)

    def get_mean(self, trainer: pl.Trainer, pl_module: VariationalNetwork) -> torch.Tensor:
        return parameters_to_vector(get_blitz_params(pl_module.model_bnn, "mu"))

    def get_var(self, trainer: pl.Trainer, pl_module: VariationalNetwork) -> torch.Tensor:
        std = parameters_to_vector(get_blitz_params(pl_module.model_bnn, "rho"))
        variance = std.exp().log1p().square() # according to https://github.com/piEsposito/blitz-bayesian-deep-learning/blob/5af11742484852c8bf69ad6fef27c230a2a0ecc2/blitz/modules/weight_sampler.py#L32
        return variance

    def get_eigen(self, trainer: pl.Trainer, pl_module: VariationalNetwork) -> torch.Tensor:
        train_dataloader = trainer.train_dataloader
        return compute_diag_hessian(pl_module.model_normal, pl_module.loss_fun, train_dataloader)


class Train_Trajectory(Callback):

    def __init__(self) -> None:
        super().__init__()
        self.train_trajectory:List[torch.Tensor] = []

    def get_parameters(self, pl_module:NeuralNetwork) -> torch.Tensor:
        raise NotImplementedError

    def on_train_epoch_start(self, trainer: "pl.Trainer", pl_module: NeuralNetwork) -> None:
        self.train_trajectory.append(self.get_parameters(pl_module).detach().clone())
        return super().on_train_epoch_start(trainer, pl_module)

    def on_train_end(self, trainer: "pl.Trainer", pl_module:NeuralNetwork) -> None:
        super().on_train_end(trainer, pl_module)
        # call append on last time, important for swa to exchange weights after training
        self.train_trajectory.append(self.get_parameters(pl_module).detach().clone())
        self.train_trajectory = torch.vstack(self.train_trajectory)

    def get_trajectory(self):
        return self.train_trajectory


class StandardTrajectoryLogger(Train_Trajectory):
    def get_parameters(self, pl_module:NeuralNetwork) -> torch.Tensor:
        return parameters_to_vector(pl_module.model.parameters())

class VariationalTrajectoryLogger(Train_Trajectory):
    def get_parameters(self, pl_module: VariationalNetwork) -> torch.Tensor:
        # print(pl_module.model)
        # print(parameters_to_vector(get_blitz_params(pl_module.model, "mu")))
        return parameters_to_vector(get_blitz_params(pl_module.model, "mu"))