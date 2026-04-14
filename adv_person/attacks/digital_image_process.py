import torch
from torch import Tensor
from torch.nn import Module

from .rand_tensor_fill import *


class DigitalImageProcess(Module):
    """Contrast, brightness and noise transforms."""

    def __init__(
        self,
        contrast: ITensorFill = ConstantTensorFill(1.0),
        brightness: ITensorFill = ConstantTensorFill(0.0),
        noise: ITensorFill = ConstantTensorFill(0.0),
    ) -> None:
        super().__init__()
        self.contrast = contrast
        self.brightness = brightness
        self.noise = noise

    def forward(self, x: Tensor):
        B = len(x)
        # Create random contrast tensor
        contrast = self.contrast.fill_(x.new_empty((B, 1, 1, 1)))
        # Create random brightness tensor
        brightness = self.brightness.fill_(x.new_empty((B, 1, 1, 1)))
        # Create random noise tensor
        noise = self.noise.fill_(torch.empty_like(x))
        x = (x - 0.5) * contrast + (brightness + 0.5) + noise
        return torch.clamp(x, 0, 1)

    def to_fixed(self):
        return DigitalImageProcess(
            ConstantTensorFill(self.contrast.expectation),
            ConstantTensorFill(self.brightness.expectation),
            ConstantTensorFill(0.0),
        )
