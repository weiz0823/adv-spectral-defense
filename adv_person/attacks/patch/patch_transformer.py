import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module

from adv_person.utils.bbox_utils import bbox_from_x1y1x2y2

from .median_pool import MedianPool2d

# Tolerance on comparison
_TOL = 1e-8


class PatchTransformer(Module):
    """PatchTransformer: transforms batch of patches

    Module providing the functionality necessary to transform a batch of patches, randomly adjusting brightness and
    contrast, adding random amount of noise, and rotating randomly. Resizes patches according to as size based on the
    batch of labels, and pads them to the dimension of an image.

    """

    kernel: Tensor

    def __init__(
        self,
        img_size: int,
        contrast: tuple[float, float] = (0.8, 1.2),
        brightness: tuple[float, float] = (-0.1, 0.1),
        angle: tuple[float, float] = (-20.0, 20.0),
        noise_factor=0.1,
        do_rotate=True,
        rand_loc=True,
        lc_scale=0.1,
        pooling="median",
        rand_sub=False,
        old_fashion=False,
        scale=0.2,
        y_ratio=1.0,
    ):
        super().__init__()
        self.min_contrast, self.max_contrast = contrast
        self.min_brightness, self.max_brightness = brightness
        self.noise_factor = noise_factor
        self.minangle, self.maxangle = map(self.degree_to_radius, angle)
        self.medianpooler = MedianPool2d(7, same=True)
        self.img_size = img_size
        self.do_rotate = do_rotate
        self.rand_loc = rand_loc
        self.lc_scale = lc_scale
        assert pooling in ("median", "avg", "gauss", "none")
        self.pooling = pooling
        self.rand_sub = rand_sub
        self.old_fashion = old_fashion
        self.scale = scale
        self.y_ratio = y_ratio

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
        self.register_buffer("kernel", kernel)
        """
        kernel = torch.cuda.FloatTensor([[0.003765, 0.015019, 0.023792, 0.015019, 0.003765],
                                         [0.015019, 0.059912, 0.094907, 0.059912, 0.015019],
                                         [0.023792, 0.094907, 0.150342, 0.094907, 0.023792],
                                         [0.015019, 0.059912, 0.094907, 0.059912, 0.015019],
                                         [0.003765, 0.015019, 0.023792, 0.015019, 0.003765]])
        self.kernel = kernel.unsqueeze(0).unsqueeze(0).expand(3,3,-1,-1)
        # It's wrong!
        """

    @staticmethod
    def degree_to_radius(x: int | float):
        return x / 180 * math.pi

    @staticmethod
    def radius_to_degree(x: int | float):
        return x / math.pi * 180

    def to_fixed_transform(self):
        contrast = (self.min_contrast + self.max_contrast) / 2
        brightness = (self.min_brightness + self.max_brightness) / 2
        angle = self.radius_to_degree((self.minangle + self.maxangle) / 2)
        return PatchTransformer(
            self.img_size,
            (contrast, contrast),
            (brightness, brightness),
            (angle, angle),
            0.0,
            self.do_rotate,
            False,
            self.lc_scale,
            self.pooling,
            False,
            self.old_fashion,
            self.scale,
            self.y_ratio,
        )

    def forward(
        self,
        adv_patch: Tensor,
        lab_batch: Tensor,
    ):
        lab_batch = bbox_from_x1y1x2y2(lab_batch, "cxcywh")
        if adv_patch.dim() == 3:
            adv_patch = adv_patch.unsqueeze(0)
        # Followings compute on cxcywh format bouding box
        SBS, _ = lab_batch.shape
        _, C, H, W = adv_patch.shape

        # add pooling
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

        # Make a batch of patches
        adv_batch = adv_patch.expand(SBS, -1, -1, -1)

        # Contrast, brightness and noise transforms

        # Create random contrast tensor
        contrast = adv_patch.new_empty((SBS, 1, 1, 1))
        if self.min_contrast < self.max_contrast:
            contrast.uniform_(self.min_contrast, self.max_contrast)
        else:
            contrast.fill_(self.min_contrast)

        # Create random brightness tensor
        brightness = adv_patch.new_empty((SBS, 1, 1, 1))
        if self.min_brightness < self.max_brightness:
            brightness.uniform_(self.min_brightness, self.max_brightness)
        else:
            brightness.fill_(self.min_brightness)

        # Create random noise tensor
        noise = adv_patch.new_empty(adv_batch.shape)
        if self.noise_factor > 0:
            noise.uniform_(-self.noise_factor, self.noise_factor)
        else:
            noise.fill_(0.0)

        # Apply contrast/brightness/noise, clamp
        adv_batch = adv_batch * contrast + brightness + noise
        adv_batch = torch.clamp(adv_batch, 0, 1)

        # Where the label class_id is 1 we don't want a patch (padding) --> fill mask with zero's
        msk_batch = adv_patch.new_ones(adv_batch.shape, dtype=torch.bool)

        # Rotation and rescaling transforms
        angle = adv_patch.new_empty(SBS)
        if self.do_rotate:
            if self.minangle < self.maxangle:
                angle.uniform_(self.minangle, self.maxangle)
            else:
                angle.fill_(self.minangle)
        else:
            angle.fill_(0.0)

        # Resizes and rotates
        # Relative scale is 0.2
        target_x = lab_batch[:, 0]
        target_y = lab_batch[:, 1]
        targetoff_x = lab_batch[:, 2]
        targetoff_y = lab_batch[:, 3]
        scale = self.scale * torch.sqrt(targetoff_x.square() + targetoff_y.square())
        if self.rand_loc:
            # Randomly move the patch
            off_x = targetoff_x * (
                adv_patch.new_empty(targetoff_x.size()).uniform_(
                    -self.lc_scale, self.lc_scale
                )
            )
            target_x = target_x + off_x
            off_y = targetoff_y * (
                adv_patch.new_empty(targetoff_y.size()).uniform_(
                    -self.lc_scale, self.lc_scale
                )
            )
            target_y = target_y + off_y

        if self.old_fashion:
            # Absolute offset
            target_y = target_y - 0.05
        else:
            # Relative offset
            target_y = target_y - 0.10 * targetoff_y

        adv_batch = adv_batch.view(SBS, C, H, W)
        msk_batch = msk_batch.view(SBS, C, H, W)

        if self.rand_sub:
            width = adv_batch.new_empty((SBS, 1)).uniform_(0.5, 1)
            height = adv_batch.new_empty((SBS, 1)).uniform_(0.8, 1)
            wst = adv_batch.new_empty((SBS, 1)).uniform_(0, 1) * (1 - width)
            hst = adv_batch.new_empty((SBS, 1)).uniform_(0, 1) * (1 - height)
            wrange = torch.arange(W, device=adv_batch.device).expand(SBS, W)
            W_msk = wrange < (wst * W)
            W_msk.logical_xor_(wrange < ((wst + width) * W))
            W_msk = W_msk.view(SBS, 1, 1, W)
            hrange = torch.arange(H, device=adv_batch.device).expand(SBS, H)
            H_msk = hrange < (hst * H)
            H_msk.logical_xor_(hrange < ((hst + height) * H))
            H_msk = H_msk.view(SBS, 1, H, 1)
            msk_batch.logical_and_(W_msk).logical_and_(H_msk)

        tx = (-target_x + 0.5) * 2
        ty = (-target_y + 0.5) * 2
        sin = torch.sin(angle)  # .to(adv_patch)
        cos = torch.cos(angle)  # .to(adv_patch)

        # Theta = rotation,rescale matrix
        x_scale = scale
        y_scale = scale * self.y_ratio
        theta = adv_patch.new_zeros(SBS, 2, 3)
        theta[:, 0, 0] = cos / x_scale
        theta[:, 0, 1] = sin / x_scale
        theta[:, 0, 2] = (tx * cos + ty * sin) / x_scale
        theta[:, 1, 0] = -sin / y_scale
        theta[:, 1, 1] = cos / y_scale
        theta[:, 1, 2] = (-tx * sin + ty * cos) / y_scale
        # theta = theta / scale.reshape(-1, 1, 1)
        # Affine transform to image space
        grid = F.affine_grid(
            theta, [SBS, C, self.img_size, self.img_size], align_corners=False
        )

        adv_batch_t = F.grid_sample(adv_batch, grid, align_corners=False)
        msk_batch_t = F.grid_sample(msk_batch.float(), grid, align_corners=False)
        adv_batch_t = torch.clamp(adv_batch_t, 0, 1)
        return adv_batch_t, msk_batch_t
