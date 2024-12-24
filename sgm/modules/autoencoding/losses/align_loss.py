import torch
import torch.nn as nn
import torch.nn.functional as F


def mean_flat(x):
    """
    Take the mean over all non-batch dimensions.
    """
    return torch.mean(x, dim=list(range(1, len(x.size()))))


class AlignLoss(nn.Module):
    def __init__(
        self,
        align_weight: float = 0.0,
        loss_type: str = "cossim",
    ):
        super().__init__()
        self.align_weight = align_weight
    
    def forward(self, latent_inputs, latent_predictions, split="train"):
        log = dict()
        latent_inputs = F.normalize(latent_inputs, dim=-1)
        latent_predictions = F.normalize(latent_predictions, dim=-1)
        loss = mean_flat(-(latent_inputs * latent_predictions).sum(dim=-1)).mean()
        loss = self.align_weight * loss
        log[f"{split}/loss/align"] = loss.mean().detach()
        return loss, log
