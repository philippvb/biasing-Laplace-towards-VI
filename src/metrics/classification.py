import torch
from torch import Tensor
from netcal.metrics import ECE

def accuracy(prediction:Tensor, target:Tensor) -> Tensor:
    return (prediction.argmax(dim=-1) == target).float().mean()

def expected_calibration_error(prediction:Tensor, target:Tensor, bins=15):
    prediction, target = prediction.cpu(), target.cpu()
    return ECE(bins=bins).measure(prediction.numpy(), target.numpy())

def nll_categorical(prediction:Tensor, target:Tensor)-> Tensor:
    dist = torch.distributions.Categorical(prediction)
    return - dist.log_prob(target).mean()

def confidence(prediction:Tensor):
    return prediction.max(dim=-1)[0].mean()


def classification_retention(probs, labels):
    maxs, argmaxs = torch.max(probs, -1)
    errs = (argmaxs != labels).float()
    uncertainties = 1.0 - maxs
    return retention_auc(errs, uncertainties)

def retention_auc(errs, uncertainties):
    retention_order = torch.argsort(uncertainties)
    # order errors by retention order
    errs = errs[retention_order]
    # sum up error and divide by N since only retained data points cause errors
    error_rates = torch.cumsum(errs, 0) / len(errs)
    # compute area under the curve (AUC)
    return error_rates.detach().mean().cpu().item()