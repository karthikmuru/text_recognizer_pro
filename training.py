import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import argparse

from model.CNN import CNN
from data.EMNIST import EMNISTDatamodule

def _setup_parser():
  parser = argparse.ArgumentParser(add_help=False)

  trainer_parser = pl.Trainer.add_argparse_args(parser)
  trainer_parser._action_groups[1].title = "Trainer Args"
  parser = argparse.ArgumentParser(add_help=False, parents=[trainer_parser])

  data_group = parser.add_argument_group("Data Args")
  EMNISTDatamodule.add_to_argparse(data_group)

  parser.add_argument("--help", action="help")

  return parser


def main():

    parser = _setup_parser()
    args = parser.parse_args()

    device = torch.device('cuda:0')
    data_module = EMNISTDatamodule(args)
    cnn = CNN()
    cnn = cnn.to(device)
    
    logger = TensorBoardLogger('logs')
    trainer = pl.Trainer.from_argparse_args(args, logger=logger)
    trainer.fit(cnn, datamodule=data_module)

if __name__ == "__main__":
    main()