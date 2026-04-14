import math
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn import Module

from adv_person.attacks.patch.median_pool import MedianPool2d


class ImageSynthesizer(Module):
    """Synthesize image from rendered model and background, with random perturbation."""

    kernel: Tensor

    def __init__(
        self,
        contrast: tuple[float, float] = (0.9, 1.1),
        brightness: tuple[float, float] = (-0.1, 0.1),
        noise_factor=0.02,
        scale: tuple[float, float] = (0.75, 1.6),
        translation: tuple[float, float] = (0.8, 1.0),
        pooling="none",
    ):
        super().__init__()
        self.min_contrast, self.max_contrast = contrast
        self.min_brightness, self.max_brightness = brightness
        self.noise_factor = noise_factor
        self.min_log_scale, self.max_log_scale = map(math.log, scale)
        self.translation_x, self.translation_y = translation
        self.medianpooler = MedianPool2d(7, same=True)
        assert pooling in ("median", "avg", "gauss", "none")
        self.pooling = pooling
        self.register_buffer("kernel", self.get_gaussian_kernel())

    def to_fixed(self):
        contrast = (self.min_contrast + self.max_contrast) / 2
        brightness = (self.min_brightness + self.max_brightness) / 2
        scale = 1.0
        return ImageSynthesizer(
            (contrast, contrast),
            (brightness, brightness),
            0.0,
            (scale, scale),
            (0.0, 0.0),
            self.pooling,
        )

    def forward(self, img_batch: Tensor, rgba_patch: Tensor):
        """Synthesize background (img_batch) with rendered model (adv_patch)."""
        B = len(img_batch)
        Ho, Wo = rgba_patch.shape[-2:]
        rgba_patch = rgba_patch[:B]  # Only front B samples

        # Compute alpha mask of the model
        patch_alpha = rgba_patch[:, -1:]
        mask = (patch_alpha > 0).to(patch_alpha)

        # Extract RGB part
        adv_patch = rgba_patch[:, :-1]

        # Add pooling
        if self.pooling == "median":
            adv_patch = self.medianpooler(adv_patch)
        elif self.pooling == "avg":
            adv_patch = F.avg_pool2d(adv_patch, 7, 3)
        elif self.pooling == "gauss":
            adv_patch = F.conv2d(adv_patch, self.kernel, padding=2)
        elif self.pooling == "none":
            pass
        else:
            raise ValueError(self.pooling)

        # Contrast, brightness and noise transforms

        # Create random contrast tensor
        contrast = adv_patch.new_empty((B, 1, 1, 1))
        if self.min_contrast < self.max_contrast:
            contrast.uniform_(self.min_contrast, self.max_contrast)
        else:
            contrast.fill_(self.min_contrast)

        # Create random brightness tensor
        brightness = adv_patch.new_empty((B, 1, 1, 1))
        if self.min_brightness < self.max_brightness:
            brightness.uniform_(self.min_brightness, self.max_brightness)
        else:
            brightness.fill_(self.min_brightness)

        # Create random noise tensor
        noise = adv_patch.new_empty(adv_patch.shape)
        if self.noise_factor > 0:
            noise.uniform_(-self.noise_factor, self.noise_factor)
        else:
            noise.fill_(0.0)

        # Apply contrast/brightness/noise, clamp
        adv_patch = adv_patch * contrast + brightness + noise
        adv_patch = torch.clamp(adv_patch, 0, 1)
        # Recover alpha channel with binarized (why?) mask
        adv_patch = torch.cat((adv_patch, mask), dim=1)

        # Logarithm uniform
        scale = adv_patch.new_empty(B)
        if self.min_log_scale < self.max_log_scale:
            scale.uniform_(self.min_log_scale, self.max_log_scale).exp_()
        else:
            scale.fill_(math.exp(self.min_log_scale))
        # This creates yxyx format bboxes
        mesh_bord = torch.stack(
            [
                torch.cat([m.nonzero().min(dim=0)[0], m.nonzero().max(dim=0)[0]])
                for m in mask.squeeze(1)
            ]
        )
        # Re-scale to [-1, 1]
        mesh_bord = (
            mesh_bord / mesh_bord.new_tensor([Ho - 1, Wo - 1, Ho - 1, Wo - 1]) * 2 - 1
        )
        pos_param = mesh_bord + mesh_bord.new_tensor([1, 1, -1, -1]) * scale.unsqueeze(
            -1
        )
        tymin, txmin, tymax, txmax = pos_param.unbind(-1)

        xdiff = (-txmax + txmin).relu()
        xmiddle = (txmax + txmin) / 2
        ydiff = (-tymax + tymin).relu()
        ymiddle = (tymax + tymin) / 2
        # assert not xdiff.allclose(xdiff.new_zeros(()))
        # assert not ydiff.allclose(ydiff.new_zeros(()))

        if self.translation_x > 0.0:
            tx = (
                xmiddle
                + torch.empty_like(txmin).uniform_(
                    -0.5 * self.translation_x, 0.5 * self.translation_x
                )
                * xdiff
            )
        else:
            tx = xmiddle
        if self.translation_y > 0.0:
            ty = (
                ymiddle
                + torch.empty_like(tymin).uniform_(
                    -0.5 * self.translation_y, 0.5 * self.translation_y
                )
                * ydiff
            )
        else:
            ty = ymiddle

        theta = adv_patch.new_zeros(B, 2, 3)
        theta[:, 0, 0] = scale
        theta[:, 0, 1] = 0
        theta[:, 0, 2] = tx
        theta[:, 1, 0] = 0
        theta[:, 1, 1] = scale
        theta[:, 1, 2] = ty

        grid = F.affine_grid(theta, img_batch.shape, align_corners=False)
        adv_batch = F.grid_sample(adv_patch, grid, align_corners=False)
        mask = adv_batch[:, -1:]
        adv_batch = adv_batch[:, :-1] * mask + img_batch * (1 - mask)
        # DEBUG
        # adv_batch = F.conv2d(adv_batch, self.kernel, padding=2)

        # This creates yxyx format bboxes
        gt = torch.stack(
            [
                torch.cat([m.nonzero().min(dim=0)[0], m.nonzero().max(dim=0)[0]])
                for m in mask.squeeze(1)
            ]
        )
        # We need relative coordinate in [0, 1] (still yxyx format)
        h, w = mask.shape[-2:]
        gt = gt / gt.new_tensor((h - 1, w - 1, h - 1, w - 1))
        # N tensors of shape (1, 4), re-arranged to xyxy format
        gt = gt[:, [1, 0, 3, 2]].unsqueeze(1).unbind(0)
        return adv_batch, gt

    def get_gaussian_kernel(self):
        # 5*5 gaussian kernel
        ksize = 5
        half = (ksize - 1) * 0.5
        sigma = 0.3 * (half - 1) + 0.8
        x = np.arange(-half, half + 1)
        x = np.exp(-np.square(x / sigma) / 2)
        x = np.outer(x, x)
        x = x / x.sum()
        x = torch.from_numpy(x).float()
        kernel = torch.zeros(3, 3, ksize, ksize)
        for i in range(3):
            kernel[i, i] = x
        return kernel
