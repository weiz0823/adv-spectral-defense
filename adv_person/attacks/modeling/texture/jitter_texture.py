from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn import Module
import torch.nn.functional as F

from adv_person.utils.typing import _pair, _size_2_t
from adv_person.models.loss import total_variation
from adv_person.attacks.patch.random_crop import PatchCropping


class JitterTexture(Module):
    """Repetitive texture generation (TCA).

    Args:
        fig_size: Output texture size.
        map_size: Original texture map size.
        color_transform: Color transformation.
    """

    def __init__(
        self,
        fig_size: _size_2_t,
        scale=1.1,
        color_transform: Module = nn.Identity(),
        map_size: Optional[_size_2_t] = None,
    ) -> None:
        super().__init__()
        self.fig_size = _pair(fig_size)
        self.map_size = int(scale * self.fig_size[0]), int(scale * self.fig_size[1])
        self.max_pos = (
            self.map_size[0] - self.fig_size[0],
            self.map_size[1] - self.fig_size[1],
        )
        if map_size is not None:
            self.map_size = _pair(map_size)
        self.color_transform = color_transform
        self.tex_map = nn.Parameter(torch.empty((1, 3) + self.map_size))
        nn.init.constant_(self.tex_map, 0.5)  # Init as gray
        self.cropping = PatchCropping(
            self.fig_size, "random", False, max_pos=self.max_pos
        )

    def forward(
        self,
        tex_map: Optional[Tensor] = None,
        transform_color=True,
        is_train=True,
    ):
        crop_pos = "random" if is_train else "center"
        self.cropping.kwargs["crop_pos"] = crop_pos
        if tex_map is None:
            tex_map = self.tex_map
        tex = self.cropping.forward(tex_map.squeeze(0)).unsqueeze(0)
        if transform_color:
            tex = self.color_transform(tex)
        return tex

    def tv_loss(self):
        return total_variation(self.tex_map).clamp_min(0.04)

    def clamp_(self, clamp_shift=0.0):
        self.tex_map.clamp_(clamp_shift, 1)
