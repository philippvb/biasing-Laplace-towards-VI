from typing import List
from src.callbacks.parameters import *
from src.callbacks.evaluation import *
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from src.callbacks.pca import *


def setup_callbacks(root_dir, callbacks:List[str], variational=False, **kwargs) -> List[Callback]:
    callback_list = []
    callback_list.append(ModelCheckpoint(dirpath=root_dir, every_n_epochs=kwargs["every_n_epochs"] if "every_n_epochs" in kwargs.keys() else None))
    for callback_name in callbacks:
        callback_name = callback_name.upper()

        if callback_name == "PARAMETERS":
            if variational:
                callback_list.append(VariationalParameterLogger(root_dir, to_wandb=not kwargs["no_log"]))
            else:
                parameter_kwargs = {}
                if "param_logging_method" in kwargs.keys():
                    parameter_kwargs["method"] = kwargs["param_logging_method"]
                callback_list.append(StandardParameterLogger(root_dir, to_wandb=not kwargs["no_log"], **parameter_kwargs))
        
        elif callback_name == "TRAJECTORY":
            if variational:
                callback_list.append(VariationalTrajectoryLogger())
            else:
                callback_list.append(StandardTrajectoryLogger())
        
        elif callback_name == "STATISTICS":
            if variational:
                callback_list.append(VariationalStatisticsCallback(n_samples=kwargs["test_n_samples"]))
            else:
                callback_list.append(StatisticsCallback())

        elif callback_name == "LAPLACE":
            if variational:
                raise ValueError("Laplace not available for Variational Networks")
            else:
                callback_list += [FitLaplace(), EvalLaplace(link_approx="mc", n_samples=100)]

        else:
            raise ValueError(f"Callback {callback_name} not implemented!")

    return callback_list

