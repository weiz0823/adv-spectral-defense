import math

import numpy as np
import pywt
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from numpy import ndarray
from PIL import Image
from torch import Tensor

CONST_RSQRT_2PI = 1 / math.sqrt(2 * math.pi)
CONST_SQRT_2 = math.sqrt(2)
AnyImageT = Tensor | ndarray | Image.Image


def arr2tensor(arr: ndarray):
    """Convert ([N,]H,W,C)-shaped np.ndarray to ([N,]C,H,W)-shaped Tensor on CPU."""
    t = torch.from_numpy(arr)
    if t.ndim == 4:
        return t.permute(0, 3, 1, 2)
    else:
        return t.permute(2, 0, 1)


def tensor2arr(t: Tensor) -> ndarray:
    """Convert ([N,]C,H,W)-shaped Tensor to ([N,]H,W,C)-shaped np.ndarray."""
    if t.ndim == 4:
        return t.permute(0, 2, 3, 1).cpu().numpy()
    else:
        return t.permute(1, 2, 0).cpu().numpy()


def image2tensor(image: AnyImageT):
    """Convert image(s) to Tensor type."""
    if isinstance(image, Image.Image):
        return TF.to_tensor(image)
    elif isinstance(image, ndarray):
        return arr2tensor(image)
    else:
        return image


def image2array(image: AnyImageT) -> ndarray:
    """Convert image(s) to np.ndarray type."""
    if isinstance(image, Image.Image):
        return np.array(image)
    elif isinstance(image, Tensor):
        return tensor2arr(image)
    else:
        return image


def compute_pad(l: int, d: int):
    """Compute padding of `l` with divisor `d`."""
    return -l % d


def pad_divisor(x: Tensor, divisor=2, value=0.5):
    """Pad 2D image(s) `x` to divisor `divisor`.

    Args:
      x: input.
      divisor: pad divisor.
      value: pad value."""
    h, w = x.shape[-2:]
    ph = compute_pad(h, divisor)
    pw = compute_pad(w, divisor)
    if ph == 0 and pw == 0:
        return x
    else:
        return F.pad(x, (0, pw, 0, ph), value=value)


def upsample(x: Tensor, factor=2):
    """Returns an upsample of image(s) `x` with factor `factor`."""
    return x.repeat_interleave(factor, dim=-2).repeat_interleave(factor, dim=-1)


def normal_pdf(x: Tensor, mu: Tensor | float, std: Tensor | float):
    """Returns the probability density function (PDF) of `x` ~ _N(mu, std^2)_."""
    return CONST_RSQRT_2PI / std * torch.exp(-0.5 * ((x - mu) / std).square())


def gaussian_kernel(
    size: int | tuple[int, int], std: float, normalize=True, device=None
):
    """Generate gaussian kernel.

    Kernels are usually small, no need to generate on GPU.

    Args:
      size: 2D kernel size.
      std: standard deviation.
      normalize: True if normalize to summation of 1.
        False if keep the original 2D Gaussian PDF value.
      device: device of the generation process.
    """
    if isinstance(size, int):
        h, w = size, size
    else:
        h, w = size
    hdist = normal_pdf(torch.arange(h, device=device), (h - 1) / 2, std)
    wdist = normal_pdf(torch.arange(w, device=device), (w - 1) / 2, std)
    x = torch.outer(hdist, wdist)
    if normalize:
        x.div_(x.sum())
    return x
