"""Typing utility, mainly on PyTorch."""

from typing import Callable, Protocol

from torch import Tensor
from torch.nn import Module


class ClfModel(Module):
    """Protocol for classification model. Input x and output logits.

    Note: inheritance of protocol is not permitted, this is a workaround."""

    def __call__(self, x: Tensor) -> Tensor: ...


class TimmModel(Module):
    """Protocol for models in timm registry."""

    def set_grad_checkpointing(self, enable: bool = ...) -> None: ...

    def get_classifier(self) -> Module:
        """Get classifier module.

        Returns:
            Module: classifier module. Typically `nn.Linear`.
        """
        ...

    def reset_classifier(self, num_classes: int, global_pool: str = ...) -> None:
        """Reset classifier.

        Args:
            num_classes (int): number of classes for the new classifier.
            global_pool (str, optional): global pooling type. Default: 'avg'.
        """
        ...


Criterion = Callable[[Tensor, Tensor], Tensor]
