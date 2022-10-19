from src.models.models import *
from src.models.basic import *
from src.models.varla import *
from src.models.variational import *
from src.models.hessian import *
from src.models.entropy_sgd import * 
from torch import nn

class ClassificationMLP(NeuralNetwork):
    def __init__(self, layer_sizes, *args, **kwargs) -> None:
        loss_fun = nn.CrossEntropyLoss()
        model = MLP(layer_sizes)
        super().__init__(model=model, loss_fun=loss_fun, *args, **kwargs)

class ClassificationSmallCNN(NeuralNetwork):
    def __init__(self, channel_sizes, image_dims, kernel_size=3, n_classes=10, *args, **kwargs) -> None:
        loss_fun = nn.CrossEntropyLoss()
        model = SmallCNN(channel_sizes, image_dims, kernel_size, n_classes)
        super().__init__(model=model, loss_fun=loss_fun, *args, **kwargs)

class ClassificationModel(NeuralNetwork):
    def __init__(self, *args, **kwargs) -> None:
        loss_fun = nn.CrossEntropyLoss()
        super().__init__(loss_fun=loss_fun, *args, **kwargs)

class ClassificationDiagLA(DiagonalVL):
    def __init__(self, *args, **kwargs) -> None:
        loss_fun = nn.CrossEntropyLoss()
        super().__init__(loss_fun=loss_fun, *args, **kwargs)

class ClassificationLastLayerDiagLA(LastLayerVariationalLaplace):
    def __init__(self, *args, **kwargs) -> None:
        loss_fun = nn.CrossEntropyLoss()
        super().__init__(loss_fun=loss_fun, *args, **kwargs)

class ClassificationDiagVB(VariationalNetwork):
    def __init__(self, *args, **kwargs) -> None:
        loss_fun = nn.CrossEntropyLoss()
        super().__init__(loss_fun=loss_fun,  *args, **kwargs)

class ClassificationEntropy(EntropyNeuralNetwork):
    def __init__(self, *args, **kwargs) -> None:
        loss_fun = nn.CrossEntropyLoss()
        super().__init__(loss_fun=loss_fun, *args, **kwargs)