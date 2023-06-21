import pytorch_lightning as pl
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import argparse


BATCH_SIZE = 64
NUM_WORKERS = 10
DATA_DIR = './data/'

class EMNIST(Dataset):
    def __init__(self, data_path : str, train : bool = False) -> None:
        
        super().__init__()
        
        self.transform = torchvision.transforms.Compose([
            lambda img: torchvision.transforms.functional.rotate(img, -90),
            lambda img: torchvision.transforms.functional.hflip(img),
            # torchvision.transforms.ToTensor()
        ])

        data = torchvision.datasets.EMNIST(root=data_path, train=train, download=False, transform=self.transform, split="byclass")
        self.x = torch.unsqueeze(data.data, dim=1).type(torch.float32)
        self.y = data.targets

    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return len(self.x)
    

class EMNISTDatamodule(pl.LightningDataModule) :
    def __init__(self, args: argparse.Namespace) -> None:
        
        super().__init__()

        self.args = vars(args) if args is not None else {}


        self.data_dir = self.args.get('data_dir', DATA_DIR)
        self.batch_size = self.args.get('batch_size', BATCH_SIZE)
        self.num_workers = self.args.get('num_workers', NUM_WORKERS)
        # self.prepare_data_per_node = False


    def setup(self, stage):
        self.mnist_train = EMNIST(data_path = self.data_dir, train=True)
        self.mnist_test = EMNIST(data_path = self.data_dir)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size, num_workers=self.num_workers)
    
    def val_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size, num_workers=self.num_workers)
    

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument(
        "--batch_size", type=int, default=BATCH_SIZE, help="Number of examples to operate on per forward step."
        )
        parser.add_argument(
        "--num_workers", type=int, default=NUM_WORKERS, help="Number of additional processes to load data."
        )
        parser.add_argument(
        "--data_dir", type=str, help="Path to the data folder.", required=True
        )
        return parser