import pytorch_lightning as pl
from torch.nn import Conv2d, Linear, MaxPool2d, ReLU, LogSoftmax
import torch.nn.functional as F
import torch
import argparse

from .base_model import BaseModel

class CNN(BaseModel):

    def __init__(self, args : argparse.Namespace = None):
        
        super().__init__(args)

        self.classes = 62

        self.conv1 = Conv2d(in_channels=1, out_channels=20, kernel_size=(5, 5))
        self.relu1 = ReLU()
        self.max_pool1 = MaxPool2d(kernel_size=(2, 2), stride=(2,2))

        self.conv2 = Conv2d(in_channels=20, out_channels=30, kernel_size=(3, 3))
        self.relu2 = ReLU()
        self.max_pool2 = MaxPool2d(kernel_size=(2, 2), stride=(2,2))

        # self.conv2 = Conv2d(in_channels=20, out_channels=20, kernel_size=(3, 3))
        # self.relu2 = ReLU()
        # self.max_pool2 = MaxPool2d(kernel_size=(2, 2), stride=(2,2))

        self.fc1 = Linear(in_features=750, out_features=1024)
        self.relu4 = ReLU()

        self.fc2 = Linear(in_features=1024, out_features=512)
        self.relu5 = ReLU()

        self.fc3 = Linear(in_features=512, out_features=self.classes)
        self.softmax = LogSoftmax(dim = 1)

    def forward(self, x : torch.Tensor) : 

        x = self.conv1(x)
        x = self.relu1(x)
        x = self.max_pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.max_pool2(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu4(x)

        x = self.fc2(x)
        x = self.relu5(x)

        x = self.fc3(x)
        x = self.softmax(x)

        return x
    
