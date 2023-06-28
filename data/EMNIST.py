import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import argparse

from .base_data_module import BaseDataModule


BATCH_SIZE = 64
NUM_WORKERS = 20
DATA_DIR = '../../data/'

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
    

class EMNISTDatamodule(BaseDataModule) :
    def __init__(self, args: argparse.Namespace) -> None:
        
        super().__init__(args)

    def setup(self, stage):
        self.train = EMNIST(data_path = self.data_dir, train=True)
        self.test = EMNIST(data_path = self.data_dir)


    