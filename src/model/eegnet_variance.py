import torch
from torch import nn, optim
import pytorch_lightning as pl
import torch.nn.functional as F
from torchmetrics.functional import accuracy
from src.model.sampling_softmax import SamplingSoftmax
from torch import Tensor
import numpy as np

class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, max_norm: int = 1, **kwargs):
        self.max_norm = max_norm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.weight.data = torch.renorm(self.weight.data, p=2, dim=0, maxnorm=self.max_norm)
        return super(Conv2dWithConstraint, self).forward(x)


class EEGNetMultiHeaded(pl.LightningModule):
    r'''
    A compact convolutional neural network (EEGNet). For more details, please refer to the following information.

    - Paper: Lawhern V J, Solon A J, Waytowich N R, et al. EEGNet: a compact convolutional neural network for EEG-based brain-computer interfaces[J]. Journal of neural engineering, 2018, 15(5): 056013.
    - URL: https://arxiv.org/abs/1611.08024
    - Related Project: https://github.com/braindecode/braindecode/tree/master/braindecode

    Below is a recommended suite for use in emotion recognition tasks:

    .. code-block:: python

        dataset = DEAPDataset(io_path=f'./deap',
                    root_path='./data_preprocessed_python',
                    online_transform=transforms.Compose([
                        transforms.To2d()
                        transforms.ToTensor(),
                    ]),
                    label_transform=transforms.Compose([
                        transforms.Select('valence'),
                        transforms.Binary(5.0),
                    ]))
        model = EEGNet(chunk_size=128,
                       num_electrodes=32,
                       dropout=0.5,
                       kernel_1=64,
                       kernel_2=16,
                       F1=8,
                       F2=16,
                       D=2,
                       num_classes=2)

    Args:
        chunk_size (int): Number of data points included in each EEG chunk, i.e., :math:`T` in the paper. (defualt: :obj:`151`)
        num_electrodes (int): The number of electrodes, i.e., :math:`C` in the paper. (defualt: :obj:`60`)
        F1 (int): The filter number of block 1, i.e., :math:`F_1` in the paper. (defualt: :obj:`8`)
        F2 (int): The filter number of block 2, i.e., :math:`F_2` in the paper. (defualt: :obj:`16`)
        D (int): The depth multiplier (number of spatial filters), i.e., :math:`D` in the paper. (defualt: :obj:`2`)
        num_classes (int): The number of classes to predict, i.e., :math:`N` in the paper. (defualt: :obj:`2`)
        kernel_1 (int): The filter size of block 1. (defualt: :obj:`64`)
        kernel_2 (int): The filter size of block 2. (defualt: :obj:`64`)
        dropout (float): Probability of an element to be zeroed in the dropout layers. (defualt: :obj:`0.25`)
    '''
    def __init__(self,
                 chunk_size: int = 151,
                 num_electrodes: int = 60,
                 F1: int = 8,
                 F2: int = 16,
                 D: int = 2,
                 num_classes: int = 2,
                 kernel_1: int = 64,
                 kernel_2: int = 16,
                 dropout: float = 0.25):
        super(EEGNetMultiHeaded, self).__init__()
        self.F1 = F1
        self.F2 = F2
        self.D = D
        self.chunk_size = chunk_size
        self.num_classes = num_classes
        self.num_electrodes = num_electrodes
        self.kernel_1 = kernel_1
        self.kernel_2 = kernel_2
        self.dropout = dropout

        self.block1 = nn.Sequential(
            nn.Conv2d(1, self.F1, (1, self.kernel_1), stride=1, padding=(0, self.kernel_1 // 2), bias=False),
            nn.BatchNorm2d(self.F1, momentum=0.01, affine=True, eps=1e-3),
            Conv2dWithConstraint(self.F1,
                                 self.F1 * self.D, (self.num_electrodes, 1),
                                 max_norm=1,
                                 stride=1,
                                 padding=(0, 0),
                                 groups=self.F1,
                                 bias=False), nn.BatchNorm2d(self.F1 * self.D, momentum=0.01, affine=True, eps=1e-3),
            nn.ELU(), nn.AvgPool2d((1, 4), stride=4), nn.Dropout(p=dropout))

        self.block2 = nn.Sequential(
            nn.Conv2d(self.F1 * self.D,
                      self.F1 * self.D, (1, self.kernel_2),
                      stride=1,
                      padding=(0, self.kernel_2 // 2),
                      bias=False,
                      groups=self.F1 * self.D),
            nn.Conv2d(self.F1 * self.D, self.F2, 1, padding=(0, 0), groups=1, bias=False, stride=1),
            nn.BatchNorm2d(self.F2, momentum=0.01, affine=True, eps=1e-3), nn.ELU(), nn.AvgPool2d((1, 8), stride=8),
            nn.Dropout(p=dropout))

        self.mean_node = nn.Linear(self.F2 * self.feature_dim, num_classes, bias=False)
        self.variance_node = nn.Sequential(
            nn.Linear(self.F2 * self.feature_dim, num_classes, bias=False),
            nn.Softplus()
        )

        self.sampling_softmax = SamplingSoftmax(num_samples = 1000)
        
    @property
    def feature_dim(self):
        with torch.no_grad():
            mock_eeg = torch.zeros(1, 1, self.num_electrodes, self.chunk_size)

            mock_eeg = self.block1(mock_eeg)
            mock_eeg = self.block2(mock_eeg)

        return mock_eeg.shape[3]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r'''
        Args:
            x (torch.Tensor): EEG signal representation, the ideal input shape is :obj:`[n, 60, 151]`. Here, :obj:`n` corresponds to the batch size, :obj:`60` corresponds to :obj:`num_electrodes`, and :obj:`151` corresponds to :obj:`chunk_size`.

        Returns:
            torch.Tensor[number of sample, number of classes]: the predicted probability that the samples belong to the classes.
        '''
        x = self.block1(x)
        x = self.block2(x)
        x = x.flatten(start_dim=1)

        variance = self.variance_node(x)
        mean = self.mean_node(x)

        mean, variance = self.sampling_softmax([mean, variance])

        return torch.stack([mean, variance])
    
    def beta_nll_loss_classification(self, mean: Tensor, variance: Tensor, target: Tensor, beta: float = 0.5):

        # Make sure mean and variance have no 0
        mean = torch.clamp(mean, 1e-6, 1 - 1e-6)
        variance = torch.clamp(variance, 1e-6, 1 - 1e-6)
        # First, calculate the normal NLL (without sum)
        log_likelihood = (target * mean.log() + (1.0 - target) * (1.0 - mean).log())

        # Calculate loss
        loss = 0.5 * variance.log() + log_likelihood / variance

        # Apply beta
        if beta > 0:
            loss = loss * variance.detach() ** beta

        #mean of both classes, and then sum over batch
        return -loss.mean(dim=-1).sum(dim=-1)
    
    
    
    def training_step(self, train_batch, batch_idx):
        x, target = train_batch
        mean, variance = self(x)

        y = F.one_hot(target, num_classes=self.num_classes)
        loss = self.beta_nll_loss_classification(mean, variance, y.float())
        self.logger.log_metrics({'train_loss': loss})
        return loss
    
    def validation_step(self, val_batch, batch_idx):
        x, target = val_batch
        mean, variance = self(x)
    
        y = F.one_hot(target, num_classes=self.num_classes)
        loss = self.beta_nll_loss_classification(mean, variance, y.float())
        self.logger.log_metrics({'val_loss': loss})
        return loss
    
    def test_step(self, test_batch, batch_idx):
        x, target = test_batch
        mean, variance = self(x)

        y = F.one_hot(target, num_classes=self.num_classes)
        loss = self.beta_nll_loss_classification(mean, variance, y.float())
        acc = accuracy(mean, y, 'binary')


        correct_var = []
        incorrect_var = []
        for i, (mean_, var_) in enumerate(zip(mean, variance)):
            if torch.argmax(mean_).item() == torch.argmax(y[i]).item():
                correct_var.append(torch.mean(var_).item())
            else:
                incorrect_var.append(torch.mean(var_).item())

        metrics = {"test_acc": acc, 
                   "test_loss": loss
                   }

        if correct_var:
            metrics['correct_var'] = np.mean(np.asarray(correct_var))
        
        if incorrect_var:
            metrics['incorrect_var'] = np.mean(np.asarray(incorrect_var))
        self.log_dict(metrics)
        return metrics

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch)

    def predict_variance(self, batch):
        return self(batch)[1]


    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=0.001, momentum=0.9)
        return optimizer