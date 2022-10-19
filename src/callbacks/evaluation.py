import torch
from torch.utils.data import DataLoader
from laplace import BaseLaplace, Laplace

from pytorch_lightning import Callback
import pytorch_lightning as pl
from typing import Any

from src.metrics.classification import *
from src.models.utils import forward_dataloader
from src.models.variational import VariationalNetwork

class FitLaplace(Callback):
    def __init__(self) -> None:
        super().__init__()

    def on_test_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        trainer.reset_train_dataloader()
        train_dataset = trainer.train_dataloader
        with torch.enable_grad():
            laplace = fit_laplace(pl_module, train_dataset)
        pl_module.laplace_model = laplace
        return super().on_test_start(trainer, pl_module)




class StatisticsCallback(Callback):
    def __init__(self, metric_prefix:str="test_") -> None:
        super().__init__()
        self.metric_prefix = metric_prefix
        self.dset_name = ""
        self.predictions = []
        self.targets = []

    def get_metric_prefix(self):
        return self.dset_name + self.metric_prefix

    def forward(self, pl_module, data, **kwargs):
        prediction = pl_module(data, **kwargs)
        return prediction.softmax(dim=-1)

    def on_test_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.dset_name = trainer.datamodule.get_name() + "_"
        self.predictions = []
        self.targets = []
        return super().on_test_epoch_start(trainer, pl_module)

    def on_test_batch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        data, target = batch
        prediction = self.forward(pl_module, data)
        self.predictions.append(prediction)
        self.targets.append(target)

    def compute_metrics(self, predictions, targets, pl_module):
        model_accuracy = accuracy(predictions, targets)
        nll = nll_categorical(predictions, targets)
        ece = expected_calibration_error(predictions, targets, bins=15)
        conf = confidence(predictions)
        ret_auc = classification_retention(predictions, targets)
        pl_module.log(self.get_metric_prefix() + "retention_auc", ret_auc, on_step=False, on_epoch=True, prog_bar=False)
        pl_module.log(self.get_metric_prefix() + "acc", model_accuracy, on_step=False, on_epoch=True, prog_bar=False)
        pl_module.log(self.get_metric_prefix() + "nll", nll, on_step=False, on_epoch=True, prog_bar=False)
        pl_module.log(self.get_metric_prefix() + "ece", ece, on_step=False, on_epoch=True, prog_bar=False)
        pl_module.log(self.get_metric_prefix() + "conf", conf, on_step=False, on_epoch=True, prog_bar=False)

    def on_test_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        # transform to torch tensor
        self.predictions = torch.cat(self.predictions)
        self.targets = torch.cat(self.targets)
        self.compute_metrics(self.predictions, self.targets, pl_module)
        return super().on_test_epoch_end(trainer, pl_module)


class OODStatistics(StatisticsCallback):
    def __init__(self, metric_prefix: str = "test_") -> None:
        super().__init__(metric_prefix)
    
    def compute_metrics(self, predictions, targets, pl_module):
        conf = confidence(predictions)
        pl_module.log(self.get_metric_prefix() + "conf", conf, on_step=False, on_epoch=True, prog_bar=False)


class VariationalStatisticsCallback(StatisticsCallback):
    def __init__(self, metric_prefix: str = "test_", n_samples=100) -> None:
        super().__init__(metric_prefix)
        self.n_samples = n_samples

    def forward(self, pl_module:VariationalNetwork, data, **kwargs):
        prediction = pl_module.predict_with_samples(data, samples=self.n_samples)
        return prediction.mean(dim=1)

class OODStatisticsVariational(VariationalStatisticsCallback):
    def __init__(self, metric_prefix: str = "test_", n_samples=100) -> None:
        super().__init__(metric_prefix, n_samples)

    def compute_metrics(self, predictions, targets, pl_module):
        conf = confidence(predictions)
        pl_module.log(self.get_metric_prefix() + "conf", conf, on_step=False, on_epoch=True, prog_bar=False)


class EvalLaplace(StatisticsCallback):
    def __init__(self, link_approx="mc", n_samples=100) -> None:
        super().__init__(metric_prefix="laplace_")
        self.n_samples=n_samples
        self.link_approx = link_approx

    def on_test_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        try:
            pl_module.laplace_model
        except AttributeError as e:
            raise AttributeError("Seems like you haven't fitted Laplace before calling Laplace eval!")
        return super().on_test_epoch_start(trainer, pl_module)

    def forward(self, pl_module, data, **kwargs):
        return pl_module.laplace_model(data, link_approx=self.link_approx, n_samples=self.n_samples)

class OODStatisticsLaplace(EvalLaplace):
    def __init__(self, link_approx="mc", n_samples=100) -> None:
        super().__init__(link_approx, n_samples)
    
    def compute_metrics(self, predictions, targets, pl_module):
        conf = confidence(predictions)
        pl_module.log(self.get_metric_prefix() + "conf", conf, on_step=False, on_epoch=True, prog_bar=False)


def fit_laplace(model, train_dataloader:DataLoader):
    print("Fitting Laplace on train dataset.")
    laplace_model = Laplace(model, likelihood="classification", hessian_structure="full")
    laplace_model.fit(train_dataloader)
    #laplace_model.optimize_prior_precision(method='marglik')
    return laplace_model

def fit_laplace_and_predict(model, train_dataloader:DataLoader, test_dataloader:DataLoader, link_approx="probit", **la_kwargs) -> tuple[torch.Tensor, torch.Tensor, BaseLaplace]:
    print("Making predictions for validation dataset.")
    laplace_model = fit_laplace(model, train_dataloader)
    predictions, targets = forward_dataloader(laplace_model, test_dataloader, link_approx=link_approx)
    return predictions, targets, laplace_model

def evaluate_statistics(prediction:torch.Tensor, target:torch.Tensor, n_classes=None):
    assert len(prediction.shape) == 2
    assert len(target.shape) == 1
    assert torch.allclose(prediction.sum(dim=-1), torch.ones(prediction.shape[0]))
    statistics = {}
    statistics["Accuracy"] = accuracy(prediction, target).item()
    statistics["ECE"] = expected_calibration_error(prediction, target, n_classes=None)
    statistics["NLL"] = nll_categorical(prediction, target).item()
    statistics["Mean confidence"] = confidence(prediction).item()
    return statistics

def compute_ood_confidence(model, ood_dataloader, **model_kwargs):
    pred_ood, _ = forward_dataloader(model, ood_dataloader, **model_kwargs)
    conf_ood = confidence(pred_ood.softmax(dim=-1))
    return conf_ood
