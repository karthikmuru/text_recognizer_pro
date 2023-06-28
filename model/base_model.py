import pytorch_lightning as pl
import argparse
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score

LEARNING_RATE = 0.0001

class BaseModel(pl.LightningModule):

    def __init__(self, args : argparse.Namespace = None):
        super().__init__()
        
        self.args = vars(args) if args is not None else {}

        self.lr = self.args.get('lr', LEARNING_RATE)

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

        self.log('val_loss', loss, prog_bar=True, on_step=False)
        self.log('val_acc', val_acc, prog_bar=True, logger=True, on_step=False)
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument(
        "--lr", type=float, default=LEARNING_RATE, help="Model Learning rate."
        )
        return parser