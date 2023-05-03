import torch
import torch.nn as nn
from torch.distributions import Normal


class SamplingSoftmax(nn.Module):
    def __init__(self, num_samples: int = 100) -> None:
        super().__init__()
        self.num_samples = num_samples

    def forward(self, inputs):
        mean_logits, variance_logits = inputs

        mean_logits_unsqueezed = torch.unsqueeze(mean_logits, 1)
        variance_logits_unsqueezed = torch.unsqueeze(variance_logits, 1)

        mean_logits_repeated = mean_logits_unsqueezed.repeat([1, self.num_samples, 1])
        variance_logits_repeated = variance_logits_unsqueezed.repeat([1, self.num_samples, 1])

        samples = torch.randn_like(mean_logits_repeated) * variance_logits_repeated + mean_logits_repeated
        samples_softmax = torch.softmax(samples, dim=-1)
        
        probability = torch.mean(samples_softmax, dim=1)
        variance = torch.var(samples_softmax, dim=1)

        return (probability, variance)
    