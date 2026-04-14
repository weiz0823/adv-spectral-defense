from typing import Optional

import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor
from torch.nn import Module


def _pad_tensor(x: Tensor, size: tuple, pos: tuple, color: tuple):
    pad_color = torch.tensor(color).reshape(-1, 1, 1) / 255
    y = x.new_empty(x.shape[:-2] + size).copy_(pad_color)
    h0, w0 = pos
    h, w = x.shape[-2:]
    y[..., h0 : h0 + h, w0 : w0 + w] = x
    return y


def _pad_pil(x: Image.Image, size: tuple, pos: tuple, color: tuple):
    padded_img = Image.new("RGB", size, color=color)
    padded_img.paste(x, pos)
    return padded_img


def pad_to_square(
    img: Image.Image | Tensor,
    bboxes: Optional[Tensor] = None,
    pad_color=(127, 127, 127),
):
    """Pad an image to square and move its bounding boxes correspondingly.

    Args:
        img: Input image
        bboxes: Input bounding boxes

    Returns:
        tuple of padded img and moved bboxes
    """
    if isinstance(img, Image.Image):
        w, h = img.size
    else:
        h, w = img.shape[-2:]
    if w == h:
        padded_img = img
    else:
        if w < h:
            padding = (h - w) // 2
            if isinstance(img, Image.Image):
                padded_img = _pad_pil(img, (h, h), (padding, 0), pad_color)
            else:
                padded_img = _pad_tensor(img, (h, h), (0, padding), pad_color)
            if bboxes is not None:
                # Move xmin, xmax
                bboxes[:, (0, 2)] = (bboxes[:, (0, 2)] * w + padding) / h
        else:
            padding = (w - h) // 2
            if isinstance(img, Image.Image):
                padded_img = _pad_pil(img, (w, w), (0, padding), pad_color)
            else:
                padded_img = _pad_tensor(img, (w, w), (padding, 0), pad_color)
            if bboxes is not None:
                # Move ymin, ymax
                bboxes[:, (1, 3)] = (bboxes[:, (1, 3)] * h + padding) / w
    if bboxes is None:
        return padded_img
    else:
        return padded_img, bboxes


class PadToSquare(Module):
    def __init__(self, pad_color=(127, 127, 127)) -> None:
        super().__init__()
        self.pad_color = pad_color

    def forward(self, img: Image.Image | Tensor, bboxes: Optional[Tensor] = None):
        return pad_to_square(img, bboxes, self.pad_color)


def pad_lab(lab: Tensor, max_n_labels: int):
    """Pad labels to maximum length."""
    pad_size = max_n_labels - lab.shape[0]
    padded_lab = F.pad(lab, (0, 0, 0, pad_size), value=-1) if pad_size > 0 else lab
    return padded_lab
