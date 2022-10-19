from pytorch_lightning.callbacks import StochasticWeightAveraging
from typing import Union, Optional, List

class SWACallback(StochasticWeightAveraging):
    def __init__(self, swa_epoch_start: Union[int, float] = 0.75, swa_lrs: float = 1e-1, annealing_epochs: int = 10, annealing_strategy: str = "cos", *arg, **kwargs):
        super().__init__(swa_epoch_start, swa_lrs, annealing_epochs, annealing_strategy)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("SWAOptimizer")

        def str_to_float_or_int(str):
            str = float(str)
            if (str % 1) == 0:
                str = int(str)
            return str

        parser.add_argument("--swa_lrs", type=float, default=1e-1)
        # those 3 were removed since now already in base class
        # parser.add_argument("--swa_epoch_start", type=str_to_float_or_int, default=0.75)
        # parser.add_argument("--annealing_strategy", type=str, default="cos")
        # parser.add_argument("--annealing_epochs", type=int, default=10)
        return parent_parser