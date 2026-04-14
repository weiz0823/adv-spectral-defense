from typing import Optional

import torch
from pytorch3d.renderer import AmbientLights, DirectionalLights, PointLights
from torch import Tensor


class LightSampler(object):
    """Sample lights from Ambient/Directional/Point light.

    This is very simple in sampling range now.
    """

    LIGHT_TYPES = ("Ambient", "Directional", "Point")

    def __init__(self, device: Optional[torch.device] = None) -> None:
        self.device = device

    def sample(self, type_ind: Optional[int] = None):
        """Sample lights.

        Args:
            type_ind: int
                Specify a type of lights. Note that theta will still be sampled.
        """
        if type_ind is None:
            type_ind = torch.randint(len(self.LIGHT_TYPES), ()).item()
            color = torch.empty((1, 3)).uniform_(0.8, 1.1)
        else:
            color = None
        theta = torch.rand(()) * (2 * torch.pi)

        # Lighting type and direction/location are sampled, but color is not
        # ambient_color=(0.85, 0.92, 1.08)
        if type_ind == 0:
            lights = AmbientLights(ambient_color=color, device=self.device)
        elif type_ind == 1:
            lights = DirectionalLights(
                device=self.device,
                direction=[[torch.sin(theta), 0.0, torch.cos(theta)]],
            )
        else:
            lights = PointLights(
                device=self.device,
                location=[[torch.sin(theta) * 3, 0.0, torch.cos(theta) * 3]],
            )
        return lights
