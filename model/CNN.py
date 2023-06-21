import pytorch_lightning as pl
from torch.nn import Conv2d, Linear, MaxPool2d, ReLU, LogSoftmax
import torch.nn.functional as F
import torch
from sklearn.metrics import accuracy_score

class CNN(pl.LightningModule):

    def __init__(self, classes : int = 62):
        
        super().__init__()

        self.conv1 = Conv2d(in_channels=1, out_channels=20, kernel_size=(5, 5))
        self.relu1 = ReLU()
        self.max_pool1 = MaxPool2d(kernel_size=(2, 2), stride=(2,2))

        self.conv2 = Conv2d(in_channels=20, out_channels=20, kernel_size=(3, 3))
        self.relu2 = ReLU()
        self.max_pool2 = MaxPool2d(kernel_size=(2, 2), stride=(2,2))

        # self.conv2 = Conv2d(in_channels=20, out_channels=20, kernel_size=(3, 3))
        # self.relu2 = ReLU()
        # self.max_pool2 = MaxPool2d(kernel_size=(2, 2), stride=(2,2))

        self.fc1 = Linear(in_features=800, out_features=1024)
        self.relu4 = ReLU()

        self.fc2 = Linear(in_features=1024, out_features=512)
        self.relu5 = ReLU()

        self.fc3 = Linear(in_features=512, out_features=classes)
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
    
    def training_step(self, batch, batch_idx) :        
        x, y = batch
        y_hat = self(x)

        _, preds = torch.max(y_hat, 1)
        
        train_acc = accuracy_score(preds.cpu(), y.cpu())

        loss = F.cross_entropy(y_hat.cpu(), y.cpu())
        
        self.log('train_acc', train_acc, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        self.log('train_loss', loss, prog_bar=False, logger=True, on_step=False, on_epoch=True)

        return {"loss" : loss}
    
    def validation_step(self, batch, batch_idx):        
        x, y = batch
        y_hat = self(x)

        _, preds = torch.max(y_hat, 1)

        loss = F.cross_entropy(y_hat, y)
        val_acc = torch.tensor(accuracy_score(preds.cpu(), y.cpu()))

        self.log('val_loss', loss, prog_bar=True)
        self.log("val_acc", val_acc, prog_bar=True, logger=True, on_step=True, on_epoch=True, )
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)
