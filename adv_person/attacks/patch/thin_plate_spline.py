from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module

from adv_person.utils.torch_utils import gen_uniform_grid
from adv_person.utils.typing import _pair, _size_2_t


class ThinPlateSpline(Module):
    """Apply Thin Plate Spline (TPS) transformation."""

    target_ctrl_points: Tensor
    mul_kernel: Tensor
    max_range: Tensor
    canvas: float

    def __init__(
        self,
        target_shape: _size_2_t,
        target_ctrl_points: Tensor,
        max_range: float | list[float] = 0.1,
        canvas=0.5,
        target_coord: Optional[Tensor] = None,
    ):
        super().__init__()
        target_shape = _pair(target_shape)
        H, W = target_shape
        self.target_shape = target_shape
        assert target_ctrl_points.ndim == 2
        N, D = target_ctrl_points.shape
        self.ndim = D
        self.register_buffer("target_ctrl_points", target_ctrl_points)
        self.canvas = canvas
        self.register_buffer("max_range", torch.tensor(max_range))

        kernel = torch.zeros(N + 1 + self.ndim, N + 1 + self.ndim)
        kernel[:N, :N].copy_(
            self.compute_partial_repr(target_ctrl_points, target_ctrl_points)
        )
        kernel[:N, N].fill_(1)
        kernel[N, :N].fill_(1)
        kernel[:N, N + 1 :].copy_(target_ctrl_points)
        kernel[N + 1 :, :N].copy_(target_ctrl_points.T)
        inv_kernel = kernel.inverse()

        if target_coord is None:
            assert self.ndim == 2
            # Get a uniform grid of target coordinates as default.
            # We get (W,H) in [-1,1]^2.
            target_coord = gen_uniform_grid((-1, -1), (1, 1), (W, H))
            # W axis changing fastest
            target_coord = target_coord.transpose(0, 1).flatten(0, -2)
        target_coord_partial = self.compute_partial_repr(
            target_coord, target_ctrl_points
        )
        target_coord_repr = torch.cat(
            [target_coord_partial, torch.ones(len(target_coord), 1), target_coord],
            dim=1,
        )
        self.register_buffer("mul_kernel", target_coord_repr @ inv_kernel)

    def get_source_coordinate(self, source_ctrl_points: Tensor):
        return self.mul_kernel @ F.pad(source_ctrl_points, (0, 0, 0, self.ndim + 1))

    def random_source_ctrl_points(self, batch_size: int):
        return (
            self.target_ctrl_points
            + self.target_ctrl_points.new_empty(
                batch_size, *self.target_ctrl_points.shape
            ).uniform_(-1, 1)
            * self.max_range
        )

    def forward(self, inputs: Tensor, source_ctrl_points: Optional[Tensor] = None):
        unsqueeze = inputs.ndim == 3
        if unsqueeze:
            inputs = inputs.unsqueeze(0)
        H, W = self.target_shape
        B = len(inputs)
        if source_ctrl_points is None:
            # Generate random source control points
            source_ctrl_points = self.random_source_ctrl_points(B)
        source_coord = self.get_source_coordinate(source_ctrl_points)
        grid = source_coord.view(B, H, W, 2)
        target_image = self.grid_sample(inputs, grid, self.canvas)
        if unsqueeze:
            target_image = target_image.squeeze(0)
        return target_image

    @staticmethod
    def compute_partial_repr(input: Tensor, control: Tensor):
        """phi(x1, x2) = r^2 * log(r), where r = ||x1 - x2||_2"""
        diff = input.unsqueeze(1) - control.unsqueeze(0)
        squared_dist = diff.square().sum(dim=-1)
        repr_matrix = 0.5 * squared_dist * torch.log(squared_dist)
        # 0*log0=0, correct nan values
        return repr_matrix.nan_to_num_(0.0, 0.0, 0.0)

    @staticmethod
    def grid_sample(input: Tensor, grid: Tensor, canvas: Optional[float] = None):
        output = F.grid_sample(input, grid, align_corners=False)
        if canvas is not None:
            input_mask = torch.ones_like(input)
            output_mask = F.grid_sample(input_mask, grid, align_corners=False)
            output = output * output_mask + canvas * (1 - output_mask)
        return output
