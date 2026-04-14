"""Contains the `Adversary` base class."""

import math
from functools import partial
from numbers import Number
from typing import Callable, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, amp, nn
from torch.nn import Module

from adv_person.utils.protocol import ClfModel, Criterion
from adv_person.utils.torch_utils import get_eps

# from .image_l0 import image_l0_norm
# from .no_sync import get_no_sync_context

# Alias for default AMP autocast to the default CUDA device
default_amp_autocast = partial(torch.autocast, device_type="cuda")


class Adversary:
    """Base class for adversaries/attacks.

    Attributes:
        eps (float):
            Attack epsilon.
        p (float | int):
            Bound the perturbation by L_p norm.
        use_amp (bool):
            Whether to use Automatic Mixed Precision (AMP).
            Default: False.
        amp_autocast (function () -> context):
            The AMP autocast function that create the autocast context.
            Default: torch autocast to CUDA device.
        grad_scaler (amp.GradScaler | None):
            Gradient scaler for AMP. Default: None.
    """

    def __init__(
        self,
        eps: float,
        p: Number = math.inf,
        use_amp=False,
        amp_autocast: Optional[partial[torch.autocast]] = None,
        grad_scaler=None,
    ):
        self.p = self._check_norm_type(p)
        if math.isinf(self.p):
            assert 0.0 <= eps and eps <= 1.0, f"Expect eps in [0,1], got {eps}"
        self.eps: float = eps
        self.use_amp = use_amp
        self.amp_autocast = amp_autocast or default_amp_autocast
        self.grad_scaler = grad_scaler

    @classmethod
    def _check_norm_type(cls, p: Number):
        """Check if the norm is valid.

        Args:
            p (Number | str):
                The norm bound.
                - Number: Valid values are {0}u[1, +inf].
                - str: Valid values are 'inf', 'fro' (converted to 2).

        Returns:
            p_converted (Number | str):
                Standard values: {0}u[1, +inf), 'inf'.

        Raises:
            ValueError: When the norm type is invalid.
        """
        if not isinstance(p, Number):
            raise ValueError(
                "Expect Number or str for norm type, use `math.inf` for infinity"
            )
        # No worries, NaN compares to False for all conditions
        if not (p == 0 or 1 <= p):
            raise ValueError(f"Expect norm type to be in {{0}}u[1, +inf], got {p}")
        return p

    def _attack(
        self, model: Module, criterion: Callable, X: Tensor, y: Optional[Tensor]
    ) -> Tensor:
        """Core function for attack implementation.

        Args:
            model (Module):
                Full model with normalization to attack.
            criterion (Callable):
                Loss function to attack.
            X ((N,C,H,W) Tensor):
                Input batch.
            y ((N,) Tensor | None):
                Batch of y_true in untargeted attack or target in targeted attack.
                Optional sometimes (e.g. with KL-divergence loss).

        Returns:
            X_adv ((N,C,H,W) Tensor):
                Adversarial input batch.

        Raises:
            NotImplementedError: The default implementation.
        """
        raise NotImplementedError

    def validate(self, X_adv: Tensor, X: Tensor, rtol=1e-5, verbose=False):
        """Validate if `X_adv` is a correct adversarial example.

        Check perturbation bound, lower bound and upper bound of `X_adv`.

        Args:
            X_adv ((...,C,H,W) Tensor): Adversarial input.
            X ((...,C,H,W) Tensor): Reference input.
            rtol (float, optional): Relative tolerance. (Default: 1e-5)

        Returns:
            bool: True if `X_adv` is correct regarding `X`.
        """
        if self.p == 0:
            # L0 norm is defined as number of perturbed pixels
            raise NotImplementedError("L0 norm is not implemented")
            perturb = image_l0_norm(X_adv - X)
        else:
            # Default is the vector norm
            # vectorize
            X_adv = X_adv.view(*X_adv.shape[:-3], -1)
            X = X.view(*X.shape[:-3], -1)
            # Built-in norm should solve all norms
            perturb: Tensor = torch.norm(X_adv - X, self.p, dim=-1)
        max_perturb = perturb.max().item()
        lower_bound = X_adv.min().item()
        upper_bound = X_adv.max().item()
        if verbose:
            print(
                f"Max perturbation: {max_perturb}, min: {lower_bound}, "
                f"max: {upper_bound}"
            )
        return (
            max_perturb <= self.eps * (1 + rtol)
            and lower_bound >= 0
            and upper_bound <= 1
        )

    @torch.enable_grad()
    def attack(self, model: Module, criterion: Callable, X: Tensor, y: Tensor):
        """Attack a batch of input.

        Args:
            model (Module):
                Full model with normalization to attack.
            criterion (Callable):
                Loss function to attack. Should always be the `reduction=none` version.
            X ((N,C,H,W) Tensor):
                Input batch.
            y ((N,) Tensor):
                Batch of y_true in untargeted attack or target in targeted attack.
                Optional sometimes (e.g. with KL-divergence loss).

        Returns:
            X_adv ((N,C,H,W) Tensor):
                Adversarial input batch.

        Implementation details:
            Here we do the wrapper work and call the core function `_attack`.
        """
        # Disable grad to accelerate
        model.requires_grad_(False)
        # Eval state should always be used to perform the attack
        training = model.training
        if training:
            model.eval()
        x_adv = self._attack(model, criterion, X, y)
        if training:
            model.train()
        model.requires_grad_()
        return x_adv

    def _get_eps_repr(self, eps=None):
        if eps is None:
            eps = self.eps
        """Get eps representation."""
        eps_disp = ""  # epsilon display
        if self.p in (0, 1):
            eps_disp = f"{eps:.0f}"
        elif math.isinf(self.p):
            eps_disp = f"{eps * 255:.0f}/255"
        else:
            eps_disp = f"{eps:.2g}"
        return eps_disp

    def __repr__(self) -> str:
        eps_disp = self._get_eps_repr()
        cls_name = self.__class__.__name__
        return "{}(eps={}, p={})".format(cls_name, eps_disp, self.p)
