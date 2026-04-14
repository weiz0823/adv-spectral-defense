import copy
from typing import Optional

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch import Tensor, nn
from torch.nn import Module

from .core import *

_SUPPORTED_PROCESS = ("blur", "mask")


def _build_pool(cfg: dict) -> Module:
    cfg = cfg.copy()  # Avoid in-place modification
    type_ = cfg.pop("type")
    assert type_ is not None, "'type' is required in pooling config"
    cls = getattr(nn, type_)
    return cls(**cfg)


class DWTPreprocessPlugin(Module):
    """Wavelet defense preprocess plugin.

    Attributes:
      level_override (int): Set to override level to consider, assign -1 to unset.
        Setting `level_override` will have another effect: cached masks will be used.
        This attribute is used for multi-level preprocessing."""

    def __init__(
        self,
        max_level=6,
        threshold=1.5 / 9,
        threshold_multiplier=2,
        process="blur",
        kern_size=33,
        pool: Optional[dict] = None,
    ) -> None:
        super().__init__()
        self.max_level = max_level
        self.threshold = threshold
        self.threshold_multiplier = threshold_multiplier
        assert process in _SUPPORTED_PROCESS, (
            f"Process should be in {_SUPPORTED_PROCESS}"
        )
        self.process = process
        # self.pool: Module | None
        if isinstance(pool, dict):
            self.pool = _build_pool(pool)
        else:
            self.pool = pool
        self.level_override = -1
        self.blur_kernel = gaussian_kernel(kern_size, kern_size)

    def calculate_masks(self, inputs: Tensor):
        H, W = inputs.shape[-2:]
        inputs = pad_divisor(inputs, 2**self.max_level)
        # Grayscale
        # TODO: can we eliminate grayscale?
        inputs_grayscale = TF.rgb_to_grayscale(inputs)
        coeffs = decode_wavelet_2d(inputs_grayscale, level=self.max_level)
        # Calculate mask for each level
        bin_masks: list[Tensor] = []
        for i in range(self.max_level):
            # Migrate the gap caused by diagonal length sqrt(2)
            norm_t = coeffs[i].norm(dim=-3, keepdim=True)
            # pooling needed
            if self.pool is not None:
                norm_t = self.pool(norm_t)
            # mask is alpha*2^k
            threshold = self.threshold * self.threshold_multiplier**i
            bin_mask: Tensor = norm_t > threshold
            bin_masks.append(bin_mask)
        # cumulative sum & upsample to original size
        bin_masks[0] = upsample(bin_masks[0], 2)
        for i in range(1, self.max_level):
            bin_masks[i] = torch.logical_or(
                bin_masks[i - 1], upsample(bin_masks[i], 2 ** (i + 1))
            )
        for i, mask in enumerate(bin_masks):
            bin_masks[i] = mask[..., :H, :W]
        return bin_masks

    def forward(self, inputs: Tensor) -> Tensor:
        """Add blurring to image regions."""
        if self.level_override == 0:
            return inputs
        elif self.level_override < 0:
            self.bin_masks = self.calculate_masks(inputs)
            level = self.max_level
        else:
            level = self.level_override
        mask = self.bin_masks[level - 1]

        if self.process == "blur":
            C = inputs.shape[-3]
            # kernel size is 2*l+1, padding should be l
            if self.blur_kernel.ndim == 2:
                # lazy init
                self.blur_kernel = self.blur_kernel.repeat(C, 1, 1, 1).to(inputs.device)
            elif self.blur_kernel.shape[0] != C:
                # auto-adjust blurring channels
                self.blur_kernel = (
                    self.blur_kernel[0].repeat(C, 1, 1, 1).to(inputs.device)
                )
            # we assume input device is always the same and don't check for device
            kernel = self.blur_kernel
            blurred = F.conv2d(inputs, kernel, padding=kernel.shape[-1] // 2, groups=C)
            inputs = torch.where(mask, blurred, inputs)
        elif self.process == "mask":
            inputs = torch.where(mask, 127.0, inputs)
        else:
            raise ValueError(self.process)
        return inputs
