import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

class DataModule(pl.LightningDataModule):
    def __init__(self, train, val, test, batch_size) -> None:
        super().__init__()
        self.train_set = train
        self.val_set = val
        self.test_val = test
        self.batch_size = batch_size

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_set, batch_size = self.batch_size, shuffle=True)
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_set, batch_size = self.batch_size, shuffle=True)
    
    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_val, batch_size = self.batch_size, shuffle=True)