from torch import Tensor
from torch.nn import Module


class ITexture(Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, **kwargs) -> Tensor:
        """Generate a texture map."""
        raise NotImplementedError

    def loss(self) -> Tensor:
        """Ctrl loss or TV loss."""
        raise NotImplementedError

    def clamp_(self, clamp_shift=0.0) -> None:
        """Clamp the texture map."""
        raise NotImplementedError
