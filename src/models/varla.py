from src.models.basic import NeuralNetwork
from torch import nn
from backpack import extend, backpack
from backpack.extensions import DiagGGNExact
import torch
from copy import deepcopy

from src.models.models import WideResNet, WideResNetLastLayer

HESSIAN_EPS_LOWER = 1e-45
HESSIAN_EPS_UPPER = 1e30

class VariationalLaplace(NeuralNetwork):
    def __init__(self, model: nn.Module, loss_fun: nn.Module, dset_size:int, tau=1, *args, **kwargs) -> None:
        extend(model)
        extend(loss_fun)
        super().__init__(model=model, loss_fun=loss_fun, *args, **kwargs)
        self.tau = tau
        self.backprop_wrapper = None
        self.automatic_optimization = False
        self.dset_size = dset_size


    def get_hparams(self):
        return super().get_hparams() + ["dset_size", "tau"]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("VariationalLaplace")
        parser.add_argument("--dset_size", type=int)
        parser.add_argument("--tau", type=float, default=1)
        parent_parser = NeuralNetwork.add_model_specific_args(parent_parser)
        return parent_parser

    def training_step(self, batch):
        opt = self.optimizers()
        opt.zero_grad()

        data, target = batch
        with self.backprop_wrapper:
            prediction = self.model(data)
            loss = self.loss_fun(prediction, target)
            self.manual_backward(loss, create_graph=True)

        hessian_loss = self.hessian_log_det()/self.dset_size
        if self.tau != 0:
            self.manual_backward(hessian_loss * self.tau)

        opt.step()

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_log_hessian", hessian_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_loss_hessian", hessian_loss * self.tau, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_loss_total", loss + hessian_loss, on_step=False, on_epoch=True, prog_bar=False)
        
        return loss
    
    def on_train_epoch_end(self) -> None:
        schedulers = self.lr_schedulers()
        schedulers.step(self.current_epoch + 1)
        return super().on_train_epoch_end()


    def hessian_log_det(self):
        raise NotImplementedError


class DiagonalVL(VariationalLaplace):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.backprop_wrapper = backpack(DiagGGNExact())
        self.isresnet = type(self.model) == WideResNet

    def hessian_log_det(self):
        diag_h_params = []
        for name, param in self.model.named_parameters():
            # exclude all the batchnorms
            if self.isresnet:
                if (not name.startswith("8")) and (len(param.shape) ==1):
                    continue
            else:
                diag_h_params.append(param.diag_ggn_exact)
        total_hessian = nn.utils.parameters_to_vector(diag_h_params)
        total_hessian = torch.clamp(total_hessian, min=HESSIAN_EPS_LOWER, max=HESSIAN_EPS_UPPER)
        log_det = total_hessian.log().mean()
        return log_det


class LastLayerVariationalLaplace(NeuralNetwork):
    def __init__(self, model: WideResNetLastLayer, loss_fun: nn.Module, dset_size:int, tau=1, *args, **kwargs) -> None:
        extend(model.linear) # just last layer
        extend(loss_fun)
        super().__init__(model=model, loss_fun=loss_fun, *args, **kwargs)
        self.tau = tau
        self.backprop_wrapper = backpack(DiagGGNExact())
        self.automatic_optimization = False
        self.dset_size = dset_size


    def get_hparams(self):
        return super().get_hparams() + ["dset_size", "tau"]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("LastLayerVariationalLaplace")
        parser.add_argument("--dset_size", type=int)
        parser.add_argument("--tau", type=int, default=1)
        parent_parser = NeuralNetwork.add_model_specific_args(parent_parser)
        return parent_parser

    def training_step(self, batch):
        opt = self.optimizers()
        opt.zero_grad()

        data, target = batch
        data = self.model.feature_extractor(data)
        with backpack(DiagGGNExact()):
            prediction = self.model.linear(data)
            loss = self.loss_fun(prediction, target)
            self.manual_backward(loss, create_graph=True)

        hessian_loss = self.hessian_log_det()/self.dset_size
        if self.tau != 0:
            self.manual_backward(hessian_loss * self.tau)

        opt.step()

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_log_hessian", hessian_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_loss_hessian", hessian_loss * self.tau, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_loss_total", loss + hessian_loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("learning_rate", self.optimizers().param_groups[0]["lr"], on_step=True, on_epoch=True, prog_bar=False)
        
        return loss
    
    def on_train_epoch_end(self) -> None:
        schedulers = self.lr_schedulers()
        schedulers.step(self.current_epoch + 1)
        return super().on_train_epoch_end()


    def hessian_log_det(self):
        return self.model.linear.weight.diag_ggn_exact.clamp(min=HESSIAN_EPS_LOWER, max=HESSIAN_EPS_UPPER).log().sum() + self.model.linear.bias.diag_ggn_exact.clamp(min=HESSIAN_EPS_LOWER, max=HESSIAN_EPS_UPPER).log().sum()
        total_hessian = nn.utils.parameters_to_vector([self.model[-1].weight.diag_ggn_exact, self.model[-1].bias.diag_ggn_exact])
        total_hessian = torch.clamp(total_hessian, min=HESSIAN_EPS_LOWER, max=HESSIAN_EPS_UPPER)
        log_det = total_hessian.log().mean()
        return log_det