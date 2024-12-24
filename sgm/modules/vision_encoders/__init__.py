import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def build_mlp(hidden_size, projector_dim, z_dim):
    return nn.Sequential(
        nn.Linear(hidden_size, projector_dim),
        nn.SiLU(),
        nn.Linear(projector_dim, projector_dim),
        nn.SiLU(),
        nn.Linear(projector_dim, z_dim),
    )


class VisionProjector(nn.Module):
    def __init__(
        self,
        image_size,
        patch_length,
        in_features,
        hidden_features=None,
        out_features=None,
        select_features=None,
    ):
        super().__init__()
        self.patch_length = patch_length
        self.select_features = select_features or in_features
        hidden_features = hidden_features or in_features
        out_features = out_features or out_features
        
        self.mlp = build_mlp(self.select_features * 4, hidden_features, out_features)
        
    def forward(self, x):
        x = x[:, :self.select_features]
        x = rearrange(x, "b c (h ph) (w pw) -> b (h w) (ph pw c)", ph=2, pw=2)
        x = self.mlp(x)
        return x
