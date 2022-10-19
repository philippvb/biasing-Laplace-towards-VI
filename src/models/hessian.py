import torch
from backpack import extend, backpack
from backpack.extensions import DiagHessian, DiagGGNExact
# from vivit.linalg.eigvalsh import EigvalshComputation
from asdfghjkl import hessian
from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader

def compute_second_order_backpack(model, loss_fun, dataloader:DataLoader, second_order_method, param_collector, use_tqdm=True):
    extend(model)
    extend(loss_fun)
    total_hessian = torch.zeros_like(nn.utils.parameters_to_vector(model.parameters()))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataloader_it = tqdm(dataloader) if use_tqdm else dataloader
    with backpack(second_order_method()):
        for (x, y) in dataloader_it:
            x, y = x.to(device), y.to(device)
            loss = loss_fun(model(x), y)
            loss.backward()
            with torch.no_grad():
                params_list = []
                for name, param in model.named_parameters():
                    params_list.append(param_collector(param))
            total_hessian += nn.utils.parameters_to_vector(params_list)
    total_hessian /= len(dataloader)
    return total_hessian

def compute_diag_hessian(model, loss_fun, dataloader, print_status=True):
    if print_status:
        print("Computing diagonal hessian with backpack")
    return compute_second_order_backpack(model, loss_fun, dataloader, DiagHessian, lambda x: x.diag_h, use_tqdm=print_status)

def compute_diag_ggn(model, loss_fun, dataloader, print_status=True):
    if print_status:
        print("Computing diag ggn with backpack")
    return compute_second_order_backpack(model, loss_fun, dataloader, DiagGGNExact, lambda x: x.diag_ggn_exact, use_tqdm=print_status)