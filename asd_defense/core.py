from .utils import *

HAAR_WAVELET = [
    [[1, 1], [1, 1]],
    [[1, 1], [-1, -1]],
    [[1, -1], [1, -1]],
    [[1, -1], [-1, 1]],
]


def decode_wavelet_2d(signal: Tensor, wavelet="db1", level=4, pad=False):
    """Decode signal(s) to a list of coefficients on each level.

    Args:
      signal (Tensor[...,H,W]): Signal(s) to decode.
      level (int): Number of coefficient levels.
      wavelet (str): Wavelet name. See PyWavelets doc. Default: 'db1'.
      pad (bool): True if pad to 2^level.

    Returns:
      coeffs (list[Tensor]): list of coefficients on each level,
        each being (...,C,H,W) tensor, from fine to coarse.
        (#dim is same as signal, except 2D signal -> 3D coeffs)
    """
    # TODO here should be a shortcut, with pre-designed wavelets such as db1, better calculate on GPU.
    if wavelet in ("haar", "db1"):
        return decode_wavelet_2d_v2_haar(signal, level, pad)
    if signal.ndim < 3:
        signal = signal.unsqueeze(-3)
    if pad:
        signal = pad_divisor(signal, 2**level)
    device = signal.device
    signal = signal.cpu().numpy()
    rv: list[Tensor] = []
    coeffs = pywt.wavedec2(signal, wavelet, level=level)
    for coeff in coeffs[:0:-1]:
        # h,w,d at each level
        rv.append(torch.from_numpy(np.concatenate(coeff, -3)).to(device))
    # cA
    rv.append(torch.from_numpy(coeffs[0]).to(device))
    return rv


def decode_wavelet_2d_v2_haar(signal: Tensor, level=4, pad=False):
    """Decode signal(s) to a list of coefficients on each level.

    Args:
      signal (Tensor[...,H,W]): Signal(s) to decode.
      level (int): Number of coefficient levels.
      wavelet (str): Wavelet name. See PyWavelets doc. Default: 'db1'.
      pad (bool): True if pad to 2^level.

    Returns:
      coeffs (list[Tensor]): list of coefficients on each level,
        each being (...,C,H,W) tensor, from fine to coarse.
        (#dim is same as signal, except 2D signal -> 3D coeffs)
    """
    kernel = signal.new_tensor(HAAR_WAVELET).reshape(4, 1, 2, 2) * 0.5
    if signal.ndim < 3:
        signal = signal.unsqueeze(-3)
    if pad:
        signal = pad_divisor(signal, 2**level)
    rv: list[Tensor] = []
    C = signal.shape[-3]
    kernel = kernel.repeat(C, 1, 1, 1)
    for i in range(level):
        signal = F.conv2d(signal, kernel, stride=2, groups=C)
        if C > 1:
            signal = (
                signal.reshape(*signal.shape[:-3], C, 4, *signal.shape[-2:])
                .transpose(-4, -3)
                .reshape(signal.shape)
            )
        rv.append(signal[..., C:, :, :])
        signal = signal[..., :C, :, :]
    rv.append(signal)
    return rv
