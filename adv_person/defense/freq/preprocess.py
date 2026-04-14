"""Frequency-based denoising preprocessing defense."""

from typing import Optional

import numpy as np
import torch
import torch.fft as fft
import torch.nn.functional as F
from scipy.fftpack import dct, idct
from torch import Tensor, nn
from torch.nn import Module


def dct2(block: np.ndarray):
    return dct(dct(block.T, norm="ortho").T, norm="ortho")


def idct2(block: np.ndarray):
    return idct(idct(block.T, norm="ortho").T, norm="ortho")


class FFTPreprocess(Module):
    """FFT-based denoising preprocessing defense.

    Attributes:
        threshold (float): Threshold for magnitude filtering.
        high_pass (bool): If True, apply high-pass filter; if False, apply low-pass filter.
        keep_ratio (float): Ratio of frequencies to keep (0-1).
    """

    def __init__(
        self,
        threshold: float = 0.1,
        high_pass: bool = False,
        keep_ratio: float = 0.8,
    ) -> None:
        super().__init__()
        self.threshold = threshold
        self.high_pass = high_pass
        self.keep_ratio = keep_ratio

    def forward(self, inputs: Tensor) -> Tensor:
        """Apply FFT-based denoising to input images.

        Args:
            inputs: Input tensor of shape (N, C, H, W).

        Returns:
            Denoised tensor of the same shape.
        """
        # Convert to complex for FFT
        inputs_complex = inputs.to(torch.complex64)

        # Apply 2D FFT
        fft_result = fft.fft2(inputs_complex, dim=(-2, -1))

        # Shift zero frequency to center
        fft_result = fft.fftshift(fft_result, dim=(-2, -1))

        # Calculate magnitude
        magnitude = torch.abs(fft_result)

        # Create mask based on threshold or keep ratio
        if self.keep_ratio < 1.0:
            # Keep only top keep_ratio frequencies
            H, W = magnitude.shape[-2:]
            total_freq = H * W
            keep_freq = int(total_freq * self.keep_ratio)

            # Flatten magnitude and get top k indices
            magnitude_flat = magnitude.view(*magnitude.shape[:-2], -1)
            _, indices = torch.topk(magnitude_flat, keep_freq, dim=-1, largest=True)

            # Create mask
            mask = torch.zeros_like(magnitude_flat, dtype=torch.bool)
            mask.scatter_(-1, indices, True)
            mask = mask.view_as(magnitude)
        else:
            # Use threshold
            mask = magnitude > self.threshold

        # Invert mask if high pass
        if self.high_pass:
            mask = ~mask

        # Apply mask
        fft_result = fft_result * mask

        # Inverse FFT
        fft_result = fft.ifftshift(fft_result, dim=(-2, -1))
        outputs = fft.ifft2(fft_result, dim=(-2, -1)).real

        # Clamp to valid range
        outputs = torch.clamp(outputs, 0, 1)

        return outputs


class DCTPreprocess(Module):
    """DCT-based denoising preprocessing defense.

    Attributes:
        threshold (float): Threshold for coefficient filtering.
        high_pass (bool): If True, apply high-pass filter; if False, apply low-pass filter.
        keep_ratio (float): Ratio of coefficients to keep (0-1).
    """

    def __init__(
        self,
        threshold: float = 0.1,
        high_pass: bool = False,
        keep_ratio: float = 0.8,
    ) -> None:
        super().__init__()
        self.threshold = threshold
        self.high_pass = high_pass
        self.keep_ratio = keep_ratio

    def _dct_2d(self, x: Tensor) -> Tensor:
        """Apply 2D DCT to input tensor."""
        # Convert to numpy array
        x_np = x.cpu().numpy()

        # Apply DCT to each element in batch and channel
        batch_size, channels, height, width = x_np.shape
        dct_result = np.zeros_like(x_np)

        for b in range(batch_size):
            for c in range(channels):
                dct_result[b, c] = dct2(x_np[b, c])

        # Convert back to tensor
        return torch.from_numpy(dct_result).to(x.device)

    def _idct_2d(self, x: Tensor) -> Tensor:
        """Apply 2D inverse DCT to input tensor."""
        # Convert to numpy array
        x_np = x.cpu().numpy()

        # Apply IDCT to each element in batch and channel
        batch_size, channels, height, width = x_np.shape
        idct_result = np.zeros_like(x_np)

        for b in range(batch_size):
            for c in range(channels):
                idct_result[b, c] = idct2(x_np[b, c])

        # Convert back to tensor
        return torch.from_numpy(idct_result).to(x.device)

    def forward(self, inputs: Tensor) -> Tensor:
        """Apply DCT-based denoising to input images.

        Args:
            inputs: Input tensor of shape (N, C, H, W).

        Returns:
            Denoised tensor of the same shape.
        """
        # Apply 2D DCT
        dct_result = self._dct_2d(inputs)

        # Calculate magnitude
        magnitude = torch.abs(dct_result)

        # Create mask based on threshold or keep ratio
        if self.keep_ratio < 1.0:
            # Keep only top keep_ratio coefficients
            H, W = magnitude.shape[-2:]
            total_coeff = H * W
            keep_coeff = int(total_coeff * self.keep_ratio)

            # Flatten magnitude and get top k indices
            magnitude_flat = magnitude.view(*magnitude.shape[:-2], -1)
            _, indices = torch.topk(magnitude_flat, keep_coeff, dim=-1, largest=True)

            # Create mask
            mask = torch.zeros_like(magnitude_flat, dtype=torch.bool)
            mask.scatter_(-1, indices, True)
            mask = mask.view_as(magnitude)
        else:
            # Use threshold
            mask = magnitude > self.threshold

        # Invert mask if high pass
        if self.high_pass:
            mask = ~mask

        # Apply mask
        dct_result = dct_result * mask

        # Inverse DCT
        outputs = self._idct_2d(dct_result)

        # Clamp to valid range
        outputs = torch.clamp(outputs, 0, 1)

        return outputs
