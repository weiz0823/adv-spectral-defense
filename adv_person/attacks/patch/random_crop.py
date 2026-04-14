from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module

from adv_person.utils import ceil_div
from adv_person.utils.typing import _pair, _size_2_t


def crop_patch(
    patch: Tensor,
    crop_size: Optional[_size_2_t] = None,
    crop_pos: str | tuple[int, int] = "center",
    circular=False,
    pad=0.0,
    max_pos=None,
    pos_shift=None,
):
    """Randomly crop a patch image.

    Args:
        patch: 3D Tensor, an image
        crop_size: pair of int | int | None
            Pair for size or int for squared size, or None for equal size.
        crop_pos: pair of int | str
            Pair for position, or string for predefined modes in 'random', 'center'.
    """
    if isinstance(crop_pos, str):
        assert crop_pos in ("center", "random")
    crop_size = _pair(crop_size)
    h, w = patch.shape[-2:]
    if crop_size is None:
        crop_size = (h, w)
    th, tw = crop_size
    rh: int
    rw: int
    if circular:
        if isinstance(crop_pos, tuple):
            rh, rw = crop_pos
        elif crop_pos == "random":
            if max_pos is not None:
                Mh, Mw = max_pos
            else:
                Mh, Mw = h, w
            rh = torch.randint(Mh, ()).item()
            rw = torch.randint(Mw, ()).item()
            if pos_shift is not None:
                rh += pos_shift[0]
                rw += pos_shift[1]
        elif crop_pos == "center":
            rh = (h - th) // 2 % h
            rw = (w - tw) // 2 % w
        # Tile the texture then crop
        eh = ceil_div(h + th, h)
        ew = ceil_div(w + tw, w)
        repeats = [1] * len(patch.shape)
        repeats[-2] = eh
        repeats[-1] = ew
        output = patch.repeat(repeats)
    else:
        if isinstance(crop_pos, tuple):
            rh, rw = crop_pos
        elif crop_pos == "random":
            if max_pos is not None:
                Mh, Mw = max_pos
            else:
                Mh, Mw = h, w
            rh = torch.randint(Mh + 1, ()).item()
            rw = torch.randint(Mw + 1, ()).item()
            if pos_shift is not None:
                rh += pos_shift[0]
                rw += pos_shift[1]
        elif crop_pos == "center":
            rh = (h - th) // 2
            rw = (w - tw) // 2
        p_left = max(0, 0 - rw)
        p_right = max(0, rw + tw - w)
        p_top = max(0, 0 - rh)
        p_bottom = max(0, rh + th - h)
        output = F.pad(patch, (p_left, p_right, p_top, p_bottom), value=pad)
        rh = max(rh, 0)
        rw = max(rw, 0)
    return output[:, rh : rh + th, rw : rw + tw]


class PatchCropping(Module):
    def __init__(
        self,
        crop_size: Optional[_size_2_t] = None,
        crop_pos: str | tuple[int, int] = "center",
        circular=False,
        pad=0.0,
        max_pos=None,
        pos_shift=None,
    ):
        super().__init__()
        self.kwargs = dict(
            crop_size=crop_size,
            crop_pos=crop_pos,
            circular=circular,
            pad=pad,
            max_pos=max_pos,
            pos_shift=pos_shift,
        )

    def forward(self, patch: Tensor):
        return crop_patch(patch, **self.kwargs)
