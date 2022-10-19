from typing import List, Optional

from pytorch_lightning import LightningDataModule
import torch
from torch.utils.data import DataLoader, Dataset, random_split, TensorDataset
from torchvision import transforms
import argparse
import numpy as np


class ToyDataSet2D(LightningDataModule):
    def __init__(self, train_size, data_noise=3, val_split:float=0., batch_size=32, num_workers=0, *args, **kwargs):
        super().__init__(None, None, None, None)

        self.name = "2DToyDataset"
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.n_classes = 1
        self.train_size = train_size
        self.data_noise = data_noise
        
        if val_split > 1:
            raise ValueError("Validation split can't be larger than 1")
        self.val_split = val_split
        self.save_hyperparameters(*self.get_hparams())



    def get_name(self):
        return self.name

    def get_hparams(self):
        return ["batch_size", "num_workers", "train_size" , "data_noise", "val_split"]

    def setup_all(self):
        self.setup("fit")
        self.setup("test")
        self.setup("predict")

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("2DToyDataset")
        parser.add_argument("--dset", type=str)
        parser.add_argument("--val_split", type=float, default=0)
        parser.add_argument("--batch_size", type=int, default=50)
        parser.add_argument("--train_size", type=int, default=50)
        parser.add_argument("--num_workers", type=int, default=0)
        parser.add_argument("--data_noise", type=int, default=3)
        return parent_parser

    def get_n_classes(self):
        return self.n_classes

    @staticmethod  
    def construct_dataset(dataset_size, data_noise):
        x = np.random.uniform(-4, 4, size=(dataset_size, 1))
        x = np.array([x for x in x if x < -2 or x > 2])
        y = ToyDataSet2D.get_y(x) + np.random.randn(*x.shape)*data_noise  # Noise is N(0, 3^2)
        x,y = torch.from_numpy(x).float(), torch.from_numpy(y).float()
        dataset = TensorDataset(x, y)
        return dataset

    @staticmethod
    def get_y(x):
        return x**3

    @staticmethod 
    def construct_test_set():
        x = np.expand_dims(np.linspace(-4, 4, 1000), axis=-1)
        y = ToyDataSet2D.get_y(x)
        x,y = torch.from_numpy(x).float(), torch.from_numpy(y).float()
        dataset = TensorDataset(x, y)
        return dataset


    def setup(self, stage: Optional[str] = None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            dset_full = ToyDataSet2D.construct_dataset(self.train_size, self.data_noise)
            n_datapoints = len(dset_full)
            if self.val_split > 0:
                train_n_datapoints = int((1-self.val_split) * n_datapoints)
                self.train_set, self.val_set = random_split(dset_full, [train_n_datapoints, n_datapoints - train_n_datapoints])

            else:
                self.train_set = dset_full
                self.val_set = ToyDataSet2D.construct_test_set()

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test_set = ToyDataSet2D.construct_test_set()

        if stage == "predict" or stage is None:
            self.predict_set = ToyDataSet2D.construct_test_set()


    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, num_workers=self.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.predict_set, batch_size=self.batch_size, num_workers=self.num_workers)