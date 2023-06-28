import pytorch_lightning as pl
from torch.utils.data import DataLoader
import argparse

BATCH_SIZE = 64
NUM_WORKERS = 20
DATA_DIR = '../../data/'

class BaseDataModule(pl.LightningDataModule):

    def __init__(self, args : argparse.Namespace = None):
        super().__init__()

        self.args = vars(args) if args is not None else {}

        self.data_dir = self.args.get('data_dir', DATA_DIR)
        self.batch_size = self.args.get('batch_size', BATCH_SIZE)
        self.num_workers = self.args.get('num_workers', NUM_WORKERS)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=self.num_workers)
    
    def val_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=self.num_workers)

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