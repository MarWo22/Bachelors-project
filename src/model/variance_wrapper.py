import torch.nn as nn
import torch


class EEGVarianceWrapper(nn.Module):
    def __init__(self, trained_model):
        super().__init__()
        self.trained_model = trained_model

    def forward(self, x: torch.Tensor):
        x = self.trained_model.block1(x)
        x = self.trained_model.block2(x)
        x = x.flatten(start_dim=1)

        variance = self.trained_model.variance_node(x)

        return variance
    
