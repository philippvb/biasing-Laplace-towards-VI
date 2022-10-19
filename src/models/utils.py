from torch import nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters
import torch
from tqdm import tqdm


def copy_params_from_bnn(bnn:nn.Module, nn:nn.Module, key="mu"):
    state_dict = parameters_to_vector(get_blitz_params(bnn, key=key))
    vector_to_parameters(state_dict, nn.parameters())


def copy_params_to_bnn(bnn:nn.Module, nn:nn.Module, key="mu"):
    state_dict = parameters_to_vector(nn.parameters()).detach().clone()
    bnn_parameters = get_blitz_params(bnn, key=key)
    bnn_samples_params = get_blitz_params(bnn, key="sampler.mu", exclude=None)
    vector_to_parameters(state_dict, bnn_parameters)
    vector_to_parameters(state_dict, bnn_samples_params)

def get_blitz_params(model:nn.Module, key:str, exclude:str="sampler") -> torch.Tensor:
    params = []
    for name, param in model.named_parameters():
        if (key in name):
            if not exclude or (not exclude in name):
                params.append(param)
    return params

def predictions_to_tensors(predictions):
    n_tensors = len(predictions[0])
    return [torch.cat([t[i] for t in predictions], axis=0) for i in range(n_tensors)]

def forward_dataloader(model_fun, dataloader, **model_kwargs):
    prediction_list, target_list = [], []
    with torch.no_grad():
        for x, target in tqdm(dataloader):
            pred = model_fun(x, **model_kwargs)
            prediction_list.append(pred)
            target_list.append(target)
    
    prediction_list = torch.cat(prediction_list, dim=0)
    target_list = torch.cat(target_list, dim=0)
    return prediction_list, target_list