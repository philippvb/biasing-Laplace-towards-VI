from src.models.basic import NeuralNetwork
from torch import nn
import torch
from torch.utils.data import DataLoader

class VariationalNetwork(NeuralNetwork):
    def __init__(self, model, model_normal:nn.Module, dset_size: int, tau=1, n_samples=1, *args, **kwargs) -> None:
        super().__init__(model=model, *args, **kwargs)
        self.n_samples = n_samples
        self.tau = tau
        self.dset_size = dset_size
        self.model_bnn = model
        self.model_normal = model_normal

    def get_hparams(self):
        return super().get_hparams() + ["tau", "n_samples", "dset_size"]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("VariationalBayes")
        parser.add_argument("--tau", type=float, default=1)
        parser.add_argument("--n_samples", type=int, default=1)
        parent_parser = NeuralNetwork.add_model_specific_args(parent_parser)
        return parent_parser

    def training_step(self, batch):
        data, target = batch
        log_lik = 0
        kl_div = 0
        for _ in range(self.n_samples):
            prediction = self.model(data)
            log_lik += self.loss_fun(prediction, target)
            kl_div += self.model.nn_kl_divergence() * self.tau / self.dset_size
        log_lik /= self.n_samples
        kl_div /= self.n_samples
        loss = log_lik + kl_div

        self.log("train_loss", log_lik, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train_kl_div", kl_div, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train_loss_total", loss, on_step=False, on_epoch=True, prog_bar=False)

        return loss

    # freezing and unfreezing for validation and test step
    def on_validation_epoch_start(self) -> None:
        # self.model.freeze_()
        return super().on_validation_epoch_start()

    def on_validation_epoch_end(self) -> None:
        # self.model.unfreeze_()
        return super().on_validation_epoch_end()

    def predict_with_samples(self, x, samples=1) -> torch.Tensor:
        prediction_list = []
        for _ in range(samples):
            prediction_list.append(self.model(x).softmax(dim=-1))
        prediction_list = torch.stack(prediction_list)
        prediction_list = prediction_list.swapaxes(0, 1) # to have samples at 0
        return prediction_list

    @torch.no_grad()
    def predict_from_dataloader(self, dataloader:DataLoader, samples=1, mean=True) -> tuple[torch.Tensor, torch.Tensor]:
        prediction_list = []
        target_list = []
        for x, target in dataloader:
            prediction = self.predict_with_samples(x, samples=samples)
            if mean:
                prediction = prediction.mean(dim=1)
            prediction_list.append(prediction)
            target_list.append(target)
        prediction_list = torch.cat(prediction_list, dim=0)
        target_list = torch.cat(target_list, dim=0)

        return prediction_list, target_list