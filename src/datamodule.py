import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np

class NestedDataSet(Dataset):
    """Dataset subclass used for the DataModule class
       Only difference with any ordinary Dataset is that this set returns a nested x value

    """

    def __init__(self, data) -> None:
        super().__init__()

        # Currently an ugly solution, since the current preprocessing method returns a list of (x, y) lists
        x_list, y_list = [], []
        for (x, y) in data:
            x_list.append(x)
            y_list.append(y)

        self.x = np.asarray(x_list)
        self.y = np.asarray(y_list)

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # X is extra nested
        return (torch.tensor([self.x[idx]]), torch.tensor(self.y[idx], dtype=torch.long))


class DataModule(pl.LightningDataModule):
    """Datamodule for the lightning trainer

    """
    def __init__(self, train, val, test, batch_size) -> None:
        super().__init__()
        self.train_set = NestedDataSet(train)
        self.val_set = NestedDataSet(val)
        self.test_set = NestedDataSet(test)
        self.batch_size = batch_size

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_set, batch_size = self.batch_size, shuffle=True)
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_set, batch_size = self.batch_size, shuffle=False)
    
    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_set, batch_size = self.batch_size, shuffle=False)
    

#Previous method
class DataModuleOld(pl.LightningDataModule):
    def __init__(self, train, val, test, batch_size) -> None:
        super().__init__()
        self.train_set = train
        self.val_set = val
        self.test_set = test
        self.batch_size = batch_size

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_set, batch_size = self.batch_size, shuffle=True)
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_set, batch_size = self.batch_size, shuffle=False)
    
    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_set, batch_size = self.batch_size, shuffle=False)