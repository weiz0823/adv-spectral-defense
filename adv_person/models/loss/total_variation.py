import torch
from torch import Tensor
from torch.nn import Module


def total_variation(image: Tensor):
    """Total variation of an image.

    Args:
        image: 3D Tensor of (C,H,W)
    """
    w_axis = (image[:, :, 1:] - image[:, :, :-1]).abs().sum()
    h_axis = (image[:, 1:, :] - image[:, :-1, :]).abs().sum()
    return (w_axis + h_axis) / image.numel()


class TotalVariation(Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor):
        return total_variation(x)
