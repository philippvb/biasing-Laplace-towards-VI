from src.datasets.torchvisiondataset import TorchVisionDataSet, NoiseDataSet
from src.datasets.toy2d import * 
from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10, FashionMNIST, CIFAR100, ImageNet

MNIST_NORMALIZATION = (0.1307,), (0.3081,)
CIFAR10_NORMALIZATION = (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)
GRAYSCALE_NORMALITAION = (0.5,), (0.5,)

class CIFAR10Dataloader(TorchVisionDataSet):
    def __init__(self, dir: str, val_split: float = 0.1, batch_size=32, num_workers=8, toy=False, additional_transforms=[], no_normalization=False, *args, **kwargs):
        dset_constructor = CIFAR10
        kwargs["image_dims"] = (3, 32, 32)
        n_classes = 10
        transform_list = [transforms.ToTensor()]
        if not no_normalization:
            transform_list.append(transforms.Normalize(*CIFAR10_NORMALIZATION))
        transform_list += additional_transforms        
        super().__init__(name="CIFAR10", dset_constructor=dset_constructor, dir=dir, val_split=val_split, batch_size=batch_size, num_workers=num_workers, toy=toy, transform=transform_list, *args, **kwargs)

class CIFAR10Noise(NoiseDataSet):
    def __init__(self, dir: str, noise:float = 1, val_split: float = 0.1, batch_size=32, num_workers=8, toy=False, additional_transforms=[], no_normalization=False, *args, **kwargs):
        dset_constructor = CIFAR10
        kwargs["image_dims"] = (3, 32, 32)
        n_classes = 10
        transform_list = [transforms.ToTensor()]
        if not no_normalization:
            transform_list.append(transforms.Normalize(*CIFAR10_NORMALIZATION))
        transform_list += additional_transforms        
        super().__init__(name="CIFAR10Noise" + str(noise), noise=noise, dset_constructor=dset_constructor, dir=dir, val_split=val_split, batch_size=batch_size, num_workers=num_workers, toy=toy, transform=transform_list, *args, **kwargs)


class CIFAR100Dataloader(TorchVisionDataSet):
    def __init__(self, dir: str, val_split: float = 0.1, batch_size=32, num_workers=8, toy=False, additional_transforms=[], no_normalization=False, *args, **kwargs):
        dset_constructor = CIFAR100
        # kwargs["image_dims"] = (3, 32, 32)
        # n_classes = 10
        transform_list = [transforms.ToTensor()]
        if not no_normalization:
            transform_list.append(transforms.Normalize(*CIFAR10_NORMALIZATION)) # leave in for now since cifar10 ood
        transform_list += additional_transforms        
        super().__init__(name="CIFAR100", dset_constructor=dset_constructor, dir=dir, val_split=val_split, batch_size=batch_size, num_workers=num_workers, toy=toy, transform=transform_list, *args, **kwargs)

class ImageNetDataloader(TorchVisionDataSet):
    def __init__(self, dir: str, val_split: float = 0.1, batch_size=32, num_workers=8, toy=False, additional_transforms=[], no_normalization=False, *args, **kwargs):
        dset_constructor = ImageNet
        # kwargs["image_dims"] = (3, 32, 32)
        # n_classes = 10
        transform_list = [transforms.ToTensor()]
        if not no_normalization:
            transform_list.append(transforms.Normalize(*CIFAR10_NORMALIZATION)) # leave in for now since cifar10 ood
        transform_list += additional_transforms        
        super().__init__(name="CIFAR100", dset_constructor=dset_constructor, dir=dir, val_split=val_split, batch_size=batch_size, num_workers=num_workers, toy=toy, transform=transform_list, *args, **kwargs)


class MNISTDataloader(TorchVisionDataSet):
    def __init__(self, dir: str, val_split: float = 0.1, batch_size=32, num_workers=8, toy=False, additional_transforms=[], *args, **kwargs):
        dset_constructor = MNIST
        kwargs["image_dims"] = (1, 28, 28)
        n_classes = 10
        transform = [transforms.ToTensor(), transforms.Normalize(*MNIST_NORMALIZATION)] + additional_transforms
        super().__init__(name="MNIST", dset_constructor=dset_constructor, dir=dir, val_split=val_split, batch_size=batch_size, num_workers=num_workers, toy=toy, transform=transform, *args, **kwargs)


class MNISTNoise(NoiseDataSet):
    def __init__(self, dir: str, noise=1, val_split: float = 0.1, batch_size=32, num_workers=8, toy=False, additional_transforms=[], *args, **kwargs):
        dset_constructor = MNIST
        kwargs["image_dims"] = (1, 28, 28)
        n_classes = 10
        transform = [transforms.ToTensor(), transforms.Normalize(*MNIST_NORMALIZATION)] + additional_transforms
        super().__init__(name="MNISTNoise", dset_constructor=dset_constructor, dir=dir, val_split=val_split, batch_size=batch_size, num_workers=num_workers, toy=toy, transform=transform, noise=noise, *args, **kwargs)

class FashionMNISTDataloader(TorchVisionDataSet):
    def __init__(self, dir: str, val_split: float = 0.1, batch_size=32, num_workers=8, toy=False, additional_transforms=[], *args, **kwargs):
        dset_constructor = FashionMNIST
        kwargs["image_dims"] = (1, 28, 28)
        n_classes = 10
        transform = [transforms.ToTensor(), transforms.Normalize(*GRAYSCALE_NORMALITAION)] + additional_transforms
        super().__init__(name="FashionMNIST", dset_constructor=dset_constructor, dir=dir, val_split=val_split, batch_size=batch_size, num_workers=num_workers, toy=toy, transform=transform, *args, **kwargs)


def get_dset(*args, **kwargs) -> TorchVisionDataSet:
    name = kwargs["dset"]
    name = name.upper()
    constructor = None
    if name == "MNIST":
        constructor = MNISTDataloader
    elif name == "FASHIONMNIST":
        constructor = FashionMNISTDataloader
    elif name == "CIFAR10":
        constructor = CIFAR10Dataloader
    elif name == "CIFAR10NOISE":
        constructor = CIFAR10Noise
    elif name == "CIFAR100":
        constructor = CIFAR100Dataloader
    elif name == "IMAGENET":
        constructor = ImageNetDataloader
    elif name == "MNISTNOISE":
        constructor = MNISTNoise
    elif name == "TOY2D":
        constructor = ToyDataSet2D
    else:
        raise ValueError(f"Dataset {name} not found!")
    return constructor(*args, **kwargs)
            

CIFAR_TRANSFORMATION = transforms.Resize(32)

def get_ood_cifar(*args, **kwargs):
    if not "additional_transforms" in kwargs.keys():
        kwargs["additional_transforms"] = []
    kwargs["additional_transforms"].append(CIFAR_TRANSFORMATION)
    return get_dset(*args, **kwargs)
