import colorsys
from typing import Callable

import torch
from torch import Tensor


def apply_color_transformation(func: Callable, x: Tensor):
    do_squeeze = x.dim() == 3
    if do_squeeze:
        x = x.unsqueeze(0)
    N, C, H, W = x.shape
    output = torch.empty_like(x)
    for n in range(N):
        for h in range(H):
            for w in range(W):
                output[n, :, h, w] = output.new_tensor(func(*x[n, :, h, w]))
    if do_squeeze:
        output = output.squeeze(0)
    return output


def rgb_to_hls(x: Tensor):
    return apply_color_transformation(colorsys.rgb_to_hls, x)


def hls_to_rgb(x: Tensor):
    return apply_color_transformation(colorsys.hls_to_rgb, x)


def rgb_to_hsv(x: Tensor):
    return apply_color_transformation(colorsys.rgb_to_hsv, x)


def hsv_to_rgb(x: Tensor):
    return apply_color_transformation(colorsys.hsv_to_rgb, x)
