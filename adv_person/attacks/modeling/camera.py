from typing import Optional

import numpy as np
import torch
from pytorch3d.renderer import FoVPerspectiveCameras, look_at_view_transform
from torch import Tensor


class CameraSampler(object):
    """Sample camera.

    Attributes:
        sampler_probs: Tensor | None
            Relative sampling probablity of each direction, used with temperature.
            Initialized to None, which implies equal probablity.
        azim: Tensor | None
            If `sample()` has been called, this variable stores last sampling result.

    Args:
        temperature: float
            (reciprocal of?) temperature when sampling camera direction.
            0.0 for exactly same probability for each direction.
    """

    def __init__(self, temperature=10.0, device: Optional[torch.device] = None) -> None:
        self.temperature = temperature  # Used with sampler_probs
        self.device = device

        # Equal probability by default
        self.sampler_probs: Optional[Tensor] = None

        # Log of last sampling
        self.azim: Optional[Tensor] = None
        self.azim_ind: Optional[Tensor] = None

    def sample(self, size=1, theta=None, elev=None, dist=None, fov=60):
        """Sample a batch of cameras.

        Args:
            size: int, batch size
            theta: _BatchFloatType, specify camera angle, None for random sampling
            elev: _BatchFloatType, specify camera elevation, None for sampling in U(2, 18)
        """
        if theta is None:
            if self.temperature > 0.0 and self.sampler_probs is not None:
                exp = (self.temperature * self.sampler_probs).softmax(dim=0)
                azim = torch.multinomial(exp, size, replacement=True)
                self.azim_ind = azim
                # Convert dtype and smoothing
                azim = azim.to(exp)
                azim = azim + torch.empty_like(azim).uniform_(-0.5, 0.5)
                self.azim = azim * (360.0 / len(exp))
            else:
                # Equal probability
                self.azim = torch.empty(size).uniform_(-180.0, 180.0)
        else:
            # Simple copy
            if True:
                self.azim = theta
            elif isinstance(theta, (float, int)):
                self.azim = torch.full(size, theta)
            elif isinstance(theta, torch.Tensor):
                self.azim = theta.clone()
            elif isinstance(theta, np.ndarray):
                self.azim = torch.from_numpy(theta)
            else:
                raise ValueError

        if elev is None:
            elev = torch.empty(size).uniform_(-5, 20)

        if dist is None:
            dist = 2.0

        R, T = look_at_view_transform(dist=dist, elev=elev, azim=self.azim)
        return FoVPerspectiveCameras(
            znear=0.5, zfar=10, device=self.device, R=R, T=T, fov=fov
        )
