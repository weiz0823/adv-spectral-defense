import math

import torch
from torch import Tensor

_EPS = 6e-8


def get_feat_norm_heatmap(feat: Tensor):
    """Get feature norm heatmap.

    Args:
        feat: 4D Tensor, with shape of BCHW.

    Returns:
        4D Tensor of B,1,H,W.
    """
    # feat = torch.clamp_min(feat.abs() - 3, 0)
    feat_norm: Tensor = feat.norm(dim=1, keepdim=True)
    avg_norm = feat_norm.mean(dim=(2, 3), keepdim=True)
    # Normalize to average over map region, 0/0=1
    feat_norm = (feat_norm + _EPS) / (avg_norm + _EPS)
    return feat_norm
