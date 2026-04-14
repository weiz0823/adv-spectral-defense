import logging
from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn import Module

from adv_person.models.loss import total_variation
from adv_person.utils.torch_utils import gen_uniform_grid_arange
from adv_person.utils.typing import _pair, _size_2_t

from .camou_generator import ctrl_loss, gumbel_color_fix_seed, prob_fix_color
from .i_texture import ITexture

_EPS = 1e-20


class CamouflageTexture(ITexture):
    """Camouflage texture managed by Voronoi diagram.

    Args:
        resolution: int, camouflage resolution.

    Potential bug:
        When size // resolution is too large (about 800),
        the Voronoi diagram becomes so large that exp(-dist/blur)->0, which causes gradient problems.
        Solution is to set `blur` larger.
    """

    colors: Tensor
    coordinates: Tensor
    seeds_fixed: Tensor

    def __init__(
        self,
        colors: list[list[float]],
        fig_size: _size_2_t,
        color_transform: Module = nn.Identity(),
        num_points=1,
        seed_ratio=0.7,
        resolution=4,
        blur=1,
    ) -> None:
        super().__init__()
        self.fig_size = _pair(fig_size)
        H, W = self.fig_size
        h, w = H // resolution, W // resolution
        self.reduced_size = (h, w)
        self.seed_ratio = seed_ratio
        self.blur = blur
        self._WARNED_PATCH_NAN_OR_INF = False

        self.color_transform = color_transform
        if not isinstance(colors, Tensor):
            colors = torch.tensor(colors)
        else:
            colors = colors.detach()
        self.register_buffer("colors", colors, False)
        num_colors = len(self.colors)
        self.register_buffer(
            "coordinates", gen_uniform_grid_arange(self.reduced_size), False
        )
        self.register_buffer("seeds_fixed", torch.empty((h, w, num_colors)).uniform_())
        self.seeds_train = nn.Parameter(torch.empty((h, w, num_colors)).uniform_())
        self.points = nn.Parameter(torch.rand((num_colors, num_points, 3)))

        k = 3
        k2 = k * k
        self.camouflage_kernel = nn.Conv2d(num_colors, num_colors, k, 1, k // 2)
        self.expand_kernel = nn.ConvTranspose2d(
            3, 3, resolution, stride=resolution, padding=0
        )
        with torch.no_grad():
            self.camouflage_kernel.weight.zero_()
            self.camouflage_kernel.bias.zero_()
            for i in range(num_colors):
                self.camouflage_kernel.weight[i, i, :, :].fill_(1 / k2)
            self.expand_kernel.weight.zero_()
            self.expand_kernel.bias.zero_()
            for i in range(3):
                self.expand_kernel.weight[i, i, :, :].fill_(1)

    def forward(
        self,
        points: Optional[Tensor] = None,
        tau=0.3,
        determinate=False,
        transform_color=True,
    ):
        if points is None:
            points = self.points
        type_ = "determinate" if determinate else "gumbel"
        prob_map = prob_fix_color(
            points, self.coordinates, self.colors, *self.reduced_size, blur=self.blur
        ).unsqueeze(0)
        prob_map = self.camouflage_kernel.forward(prob_map)
        prob_map = prob_map.squeeze(0).permute(1, 2, 0)

        seeds = (
            self.seed_ratio * self.seeds_train
            + (1 - self.seed_ratio) * self.seeds_fixed
        )
        gb = -(-(seeds + _EPS).log() + _EPS).log()
        tex = gumbel_color_fix_seed(prob_map, gb, self.colors, tau=tau, type=type_)
        tex = tex.permute(0, 3, 1, 2)  # NHWC -> NCHW
        if transform_color:
            tex = self.color_transform(tex)
        tex = self.expand_kernel.forward(tex)
        # DEBUG
        # max_pos = int(0.1 * self.fig_size[0]), int(0.1 * self.fig_size[1])
        # from adv_person.attacks.patch import crop_patch
        # tex = crop_patch(tex.squeeze(0), self.fig_size, "random", max_pos=max_pos).unsqueeze(0)
        return tex

    def ctrl_loss(self):
        return ctrl_loss(self.points, *self.fig_size)

    def tv_loss(self):
        tex = self.forward(determinate=True)
        return total_variation(tex).clamp_min(0.04)

    def loss(self):
        return self.ctrl_loss()

    def clamp_(self, clamp_shift=0.0):
        if not self._WARNED_PATCH_NAN_OR_INF and not (
            torch.all(torch.isfinite(self.seeds_train))
            and torch.all(torch.isfinite(self.points))
        ):
            logging.warning("NaN or Inf found in camouflage params.")
            self._WARNED_PATCH_NAN_OR_INF = True
        self.seeds_train.nan_to_num_(0, 0, 0)
        self.points.nan_to_num_(0, 0, 0)
        self.seeds_train.clamp_(clamp_shift, 1 - clamp_shift)
        self.points.clamp_(0, 1)
