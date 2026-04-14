import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn import Module


class ITensorFill(object):
    def fill_(self, x: Tensor) -> Tensor:
        raise NotImplementedError


class ConstantTensorFill(ITensorFill):
    def __init__(self, val) -> None:
        super().__init__()
        self.val = val
        self.expectation = val

    def fill_(self, x: Tensor):
        return x.fill_(self.val)


class RandomTensorFill(ITensorFill):
    def __init__(self, min=0, max=1) -> None:
        super().__init__()
        self.min = min
        self.max = max
        self.expectation = (min + max) / 2

    def fill_(self, x: Tensor):
        if self.min < self.max:
            return x.uniform_(self.min, self.max)
        else:
            return x.fill_(self.min)


class NormalTensorFill(ITensorFill):
    def __init__(self, mean=0, std=1, min=None, max=None) -> None:
        super().__init__()
        self.mean = mean
        self.std = std
        self.min = min
        self.max = max
        self.expectation = mean

    def fill_(self, x: Tensor):
        if self.std > 0:
            x.normal_(self.mean, self.std)
            if self.min is not None or self.max is not None:
                x.clamp_(self.min, self.max)
            return x
        else:
            return x.fill_(self.mean)
