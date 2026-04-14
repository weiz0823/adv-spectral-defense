import logging
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn import Module

from adv_person.models.loss import total_variation
from adv_person.utils.typing import _pair, _size_2_t

from .i_texture import ITexture


class SimpleTexture(ITexture):
    """Simple texture generation.

    Args:
        fig_size: Output texture size.
        map_size: Original texture map size. If None, set equal to fig_size.
        color_transform: Color transformation.
    """

    def __init__(
        self,
        fig_size: _size_2_t,
        map_size: Optional[_size_2_t] = None,
        color_transform: Module = nn.Identity(),
    ) -> None:
        super().__init__()
        self.fig_size = _pair(fig_size)
        if map_size is None:
            self.map_size = self.fig_size
            self.do_interpolate = False
        else:
            self.map_size = _pair(map_size)
            self.do_interpolate = True
        self.color_transform = color_transform
        self.tex_map = nn.Parameter(torch.empty((1, 3) + self.map_size))
        # nn.init.constant_(self.tex_map, 0.5)  # Init as gray
        self._WARNED_PATCH_NAN_OR_INF = False

    def forward(
        self,
        tex_map: Optional[Tensor] = None,
        transform_color=True,
    ):
        if tex_map is None:
            tex_map = self.tex_map
        if self.do_interpolate:
            tex = F.interpolate(tex_map, self.fig_size)
        else:
            tex = tex_map
        if transform_color:
            tex = self.color_transform(tex)
        return tex

    def tv_loss(self):
        return total_variation(self.tex_map).clamp_min(0.04)

    def loss(self):
        return self.tv_loss()

    def clamp_(self, clamp_shift=0.0):
        if not self._WARNED_PATCH_NAN_OR_INF and not torch.all(
            torch.isfinite(self.tex_map)
        ):
            logging.warning("NaN or Inf found in texture.")
            self._WARNED_PATCH_NAN_OR_INF = True
        self.tex_map.nan_to_num_(0, 0, 0)
        self.tex_map.clamp_(clamp_shift, 1 - clamp_shift)
