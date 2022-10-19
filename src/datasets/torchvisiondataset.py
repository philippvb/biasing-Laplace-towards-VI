from typing import Optional

from pytorch_lightning import LightningDataModule
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
import argparse


class TorchVisionDataSet(LightningDataModule):
    def __init__(self, name:str, dset_constructor:Dataset, dir:str, image_dims, n_classes:int, val_split:float=0.1, batch_size=32, num_workers=8, toy=False, transform=[transforms.ToTensor()], augment_dset=False, *args, **kwargs):
        super().__init__(None, None, None, None)
        self.name = name
        self.dir = dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.toy = toy
         # to be similar to agustinus bayesian ood code
        self.train_transform = transforms.Compose(transform) if type(transform) == list else transform
        self.test_transform = transforms.Compose(transform) if type(transform) == list else transform
        if augment_dset:
            transform_list_augment = transform + [transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(image_dims[-1], padding=4, padding_mode='reflect')]
            self.train_transform = transforms.RandomChoice([transforms.Compose(self.train_transform), transforms.Compose(transform_list_augment)])
    
        self.dset_constructor = dset_constructor
        self.image_dims = image_dims
        self.n_classes = n_classes
        
        if val_split > 1:
            raise ValueError("Validation split can't be larger than 1")
        self.val_split = val_split
        self.save_hyperparameters(*self.get_hparams())

    def get_name(self):
        return self.name

    def get_hparams(self):
        return ["dir", "batch_size", "num_workers", "image_dims", "n_classes"]

    def setup_all(self):
        self.setup("fit")
        self.setup("test")
        self.setup("predict")

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("TorchvisionDataSet")
        parser.add_argument("--dset", type=str)
        parser.add_argument("--no_normalization", action=argparse.BooleanOptionalAction, default=False)
        parser.add_argument("--augment_dset", action=argparse.BooleanOptionalAction, default=False)
        parser.add_argument("--dir", type=str)
        parser.add_argument("--val_split", type=float, default=0.1)
        parser.add_argument("--batch_size", type=int, default=32)
        parser.add_argument("--num_workers", type=int, default=8)
        parser.add_argument('--toy', action=argparse.BooleanOptionalAction, default=False)
        return parent_parser

    def get_image_dims(self):
        return self.image_dims

    def get_image_channels(self):
        return self.image_dims[0]
    
    def get_n_classes(self):
        return self.n_classes

    def prepare_data(self):
        # download
        self.dset_constructor(self.dir, train=True, download=True)
        self.dset_constructor(self.dir, train=False, download=True)

    def split_toy(self, dset, n_points):
        return random_split(dset, [n_points, len(dset) - n_points])[0]

    def setup(self, stage: Optional[str] = None):

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            dset_full = self.dset_constructor(self.dir, train=True, transform=self.train_transform)
            n_datapoints = len(dset_full)
            if self.toy:
                self.train_set, self.val_set, _ = random_split(dset_full, [2 * self.batch_size, 2 * self.batch_size, n_datapoints - 4 * self.batch_size])
            else:
                train_n_datapoints = int((1-self.val_split) * n_datapoints)
                self.train_set, self.val_set = random_split(dset_full, [train_n_datapoints, n_datapoints - train_n_datapoints], generator=torch.Generator().manual_seed(42))

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test_set = self.dset_constructor(self.dir, train=False, transform=self.test_transform)
            if self.toy:
                self.test_set = self.split_toy(self.test_set, 2 * self.batch_size)

        if stage == "predict" or stage is None:
            self.predict_set = self.dset_constructor(self.dir, train=False, transform=self.test_transform)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, num_workers=self.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.predict_set, batch_size=self.batch_size, num_workers=self.num_workers)


class NoiseDataSet(TorchVisionDataSet):
    def __init__(self, dset_constructor: Dataset, dir: str, image_dims, n_classes: int, noise:float, val_split: float = 0.1, batch_size=32, num_workers=8, toy=False, transform=[transforms.ToTensor()], *args, **kwargs):
        def add_gaussian(tensor):
            return tensor + torch.randn_like(tensor) * noise
        transform.append(add_gaussian)
        super().__init__(dset_constructor=dset_constructor, dir=dir, image_dims=image_dims, n_classes=n_classes, val_split=val_split, batch_size=batch_size, num_workers=num_workers, toy=toy, transform=transform, *args, **kwargs)
    
