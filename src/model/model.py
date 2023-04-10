from torch import nn, optim
from torch.nn import functional as F
import pytorch_lightning as pl
from torcheeg.models import EEGNet

class Model(pl.LightningModule):
    
    def __init__(self) -> None:
        super().__init__()

        self.model = EEGNet(chunk_size=128,
               num_electrodes=64,
               dropout=0.5,
               kernel_1=64,
               kernel_2=16,
               F1=8,
               F2=16,
               D=2,
               num_classes=2)
    
    def forward(self, x):
        return self.model(x)

    def cross_entropy_loss(self, logits, labels):
        return F.nll_loss(logits, labels)
    
    def training_step(self, train_batch, batch_id):
        x, y = train_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        self.log('train loss', loss)
        return loss
    
    def validation_step(self, val_batch, batch_id):
        x, y = val_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        self.log('validation loss', loss)
        return loss
    
    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=0.001, momentum=0.9)
        return optimizer
