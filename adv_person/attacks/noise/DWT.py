import math

import numpy as np
import pywt
import torch
from torch.autograd import Function
from torch.nn import Module


class DWTFunction_2D_tiny(Function):
    def forward(ctx, input, matrix_Low_0, matrix_Low_1, matrix_High_0, matrix_High_1):
        ctx.save_for_backward(matrix_Low_0, matrix_Low_1, matrix_High_0, matrix_High_1)
        L = torch.matmul(matrix_Low_0, input)
        LL = torch.matmul(L, matrix_Low_1)
        return LL

    def backward(ctx, grad_LL):
        matrix_Low_0, matrix_Low_1, matrix_High_0, matrix_High_1 = ctx.saved_tensors
        grad_L = torch.matmul(grad_LL, matrix_Low_1.t())
        grad_input = torch.matmul(matrix_Low_0.t(), grad_L)
        return grad_input, None, None, None, None


class IDWT_2D_tiny(Module):
    def __init__(self, wavename):
        super(IDWT_2D_tiny, self).__init__()
        wavelet = pywt.Wavelet(wavename)
        self.band_low = wavelet.dec_lo
        self.band_low.reverse()
        self.band_high = wavelet.dec_hi
        self.band_high.reverse()
        assert len(self.band_low) == len(self.band_high)
        self.band_length = len(self.band_low)
        assert self.band_length % 2 == 0
        self.band_length_half = math.floor(self.band_length / 2)

    def get_matrix(self):
        L1 = np.max((self.input_height, self.input_width))
        L = math.floor(L1 / 2)
        matrix_h = np.zeros((L, L1 + self.band_length - 2))
        matrix_g = np.zeros((L1 - L, L1 + self.band_length - 2))
        end = None if self.band_length_half == 1 else (-self.band_length_half + 1)
        index = 0
        for i in range(L):
            for j in range(self.band_length):
                matrix_h[i, index + j] = self.band_low[j]
            index += 2
        matrix_h_0 = matrix_h[
            0 : (math.floor(self.input_height / 2)),
            0 : (self.input_height + self.band_length - 2),
        ]
        matrix_h_1 = matrix_h[
            0 : (math.floor(self.input_width / 2)),
            0 : (self.input_width + self.band_length - 2),
        ]
        index = 0
        for i in range(L1 - L):
            for j in range(self.band_length):
                matrix_g[i, index + j] = self.band_high[j]
            index += 2
        matrix_g_0 = matrix_g[
            0 : (self.input_height - math.floor(self.input_height / 2)),
            0 : (self.input_height + self.band_length - 2),
        ]
        matrix_g_1 = matrix_g[
            0 : (self.input_width - math.floor(self.input_width / 2)),
            0 : (self.input_width + self.band_length - 2),
        ]
        matrix_h_0 = matrix_h_0[:, (self.band_length_half - 1) : end]
        matrix_h_1 = matrix_h_1[:, (self.band_length_half - 1) : end]
        matrix_h_1 = np.transpose(matrix_h_1)
        matrix_g_0 = matrix_g_0[:, (self.band_length_half - 1) : end]
        matrix_g_1 = matrix_g_1[:, (self.band_length_half - 1) : end]
        matrix_g_1 = np.transpose(matrix_g_1)
        if torch.cuda.is_available():
            self.matrix_low_0 = torch.Tensor(matrix_h_0).cuda()
            self.matrix_low_1 = torch.Tensor(matrix_h_1).cuda()
            self.matrix_high_0 = torch.Tensor(matrix_g_0).cuda()
            self.matrix_high_1 = torch.Tensor(matrix_g_1).cuda()
        else:
            self.matrix_low_0 = torch.Tensor(matrix_h_0)
            self.matrix_low_1 = torch.Tensor(matrix_h_1)
            self.matrix_high_0 = torch.Tensor(matrix_g_0)
            self.matrix_high_1 = torch.Tensor(matrix_g_1)

    def forward(self, LL):
        assert len(LL.size()) == 4
        self.input_height = LL.size()[-2] * 2
        self.input_width = LL.size()[-1] * 2
        self.get_matrix()
        return IDWTFunction_2D_tiny.apply(LL, self.matrix_low_0, self.matrix_low_1)


class DWT_2D_tiny(Module):
    def __init__(self, wavename):
        super(DWT_2D_tiny, self).__init__()
        wavelet = pywt.Wavelet(wavename)
        self.band_low = wavelet.rec_lo
        self.band_high = wavelet.rec_hi
        assert len(self.band_low) == len(self.band_high)
        self.band_length = len(self.band_low)
        assert self.band_length % 2 == 0
        self.band_length_half = math.floor(self.band_length / 2)

    def get_matrix(self):
        L1 = np.max((self.input_height, self.input_width))
        L = math.floor(L1 / 2)
        matrix_h = np.zeros((L, L1 + self.band_length - 2))
        matrix_g = np.zeros((L1 - L, L1 + self.band_length - 2))
        end = None if self.band_length_half == 1 else (-self.band_length_half + 1)
        index = 0
        for i in range(L):
            for j in range(self.band_length):
                matrix_h[i, index + j] = self.band_low[j]
            index += 2
        matrix_h_0 = matrix_h[
            0 : (math.floor(self.input_height / 2)),
            0 : (self.input_height + self.band_length - 2),
        ]
        matrix_h_1 = matrix_h[
            0 : (math.floor(self.input_width / 2)),
            0 : (self.input_width + self.band_length - 2),
        ]
        index = 0
        for i in range(L1 - L):
            for j in range(self.band_length):
                matrix_g[i, index + j] = self.band_high[j]
            index += 2
        matrix_g_0 = matrix_g[
            0 : (self.input_height - math.floor(self.input_height / 2)),
            0 : (self.input_height + self.band_length - 2),
        ]
        matrix_g_1 = matrix_g[
            0 : (self.input_width - math.floor(self.input_width / 2)),
            0 : (self.input_width + self.band_length - 2),
        ]
        matrix_h_0 = matrix_h_0[:, (self.band_length_half - 1) : end]
        matrix_h_1 = matrix_h_1[:, (self.band_length_half - 1) : end]
        matrix_h_1 = np.transpose(matrix_h_1)
        matrix_g_0 = matrix_g_0[:, (self.band_length_half - 1) : end]
        matrix_g_1 = matrix_g_1[:, (self.band_length_half - 1) : end]
        matrix_g_1 = np.transpose(matrix_g_1)
        if torch.cuda.is_available():
            self.matrix_low_0 = torch.Tensor(matrix_h_0).cuda()
            self.matrix_low_1 = torch.Tensor(matrix_h_1).cuda()
            self.matrix_high_0 = torch.Tensor(matrix_g_0).cuda()
            self.matrix_high_1 = torch.Tensor(matrix_g_1).cuda()
        else:
            self.matrix_low_0 = torch.Tensor(matrix_h_0)
            self.matrix_low_1 = torch.Tensor(matrix_h_1)
            self.matrix_high_0 = torch.Tensor(matrix_g_0)
            self.matrix_high_1 = torch.Tensor(matrix_g_1)

    def forward(self, input):
        assert len(input.size()) == 4
        self.input_height = input.size()[-2]
        self.input_width = input.size()[-1]
        self.get_matrix()
        return DWTFunction_2D_tiny.apply(
            input,
            self.matrix_low_0,
            self.matrix_low_1,
            self.matrix_high_0,
            self.matrix_high_1,
        )


class IDWTFunction_2D_tiny(Function):
    def forward(ctx, input_LL, matrix_Low_0, matrix_Low_1):
        ctx.save_for_backward(matrix_Low_0, matrix_Low_1)
        L = torch.matmul(input_LL, matrix_Low_1.t())
        output = torch.matmul(matrix_Low_0.t(), L)
        return output

    def backward(ctx, grad_output):
        matrix_Low_0, matrix_Low_1 = ctx.saved_tensors
        grad_L = torch.matmul(matrix_Low_0, grad_output)
        grad_LL = torch.matmul(grad_L, matrix_Low_1)
        return grad_LL, None, None, None, None


class DWT_2D(Module):
    def __init__(self, wavename):
        super(DWT_2D, self).__init__()
        wavelet = pywt.Wavelet(wavename)
        self.band_low = wavelet.rec_lo
        self.band_high = wavelet.rec_hi
        assert len(self.band_low) == len(self.band_high)
        self.band_length = len(self.band_low)
        assert self.band_length % 2 == 0
        self.band_length_half = math.floor(self.band_length / 2)

    def get_matrix(self):
        L1 = np.max((self.input_height, self.input_width))
        L = math.floor(L1 / 2)
        matrix_h = np.zeros((L, L1 + self.band_length - 2))
        matrix_g = np.zeros((L1 - L, L1 + self.band_length - 2))
        end = None if self.band_length_half == 1 else (-self.band_length_half + 1)
        index = 0
        for i in range(L):
            for j in range(self.band_length):
                matrix_h[i, index + j] = self.band_low[j]
            index += 2
        matrix_h_0 = matrix_h[
            0 : (math.floor(self.input_height / 2)),
            0 : (self.input_height + self.band_length - 2),
        ]
        matrix_h_1 = matrix_h[
            0 : (math.floor(self.input_width / 2)),
            0 : (self.input_width + self.band_length - 2),
        ]
        index = 0
        for i in range(L1 - L):
            for j in range(self.band_length):
                matrix_g[i, index + j] = self.band_high[j]
            index += 2
        matrix_g_0 = matrix_g[
            0 : (self.input_height - math.floor(self.input_height / 2)),
            0 : (self.input_height + self.band_length - 2),
        ]
        matrix_g_1 = matrix_g[
            0 : (self.input_width - math.floor(self.input_width / 2)),
            0 : (self.input_width + self.band_length - 2),
        ]
        matrix_h_0 = matrix_h_0[:, (self.band_length_half - 1) : end]
        matrix_h_1 = matrix_h_1[:, (self.band_length_half - 1) : end]
        matrix_h_1 = np.transpose(matrix_h_1)
        matrix_g_0 = matrix_g_0[:, (self.band_length_half - 1) : end]
        matrix_g_1 = matrix_g_1[:, (self.band_length_half - 1) : end]
        matrix_g_1 = np.transpose(matrix_g_1)
        if torch.cuda.is_available():
            self.matrix_low_0 = torch.Tensor(matrix_h_0).cuda()
            self.matrix_low_1 = torch.Tensor(matrix_h_1).cuda()
            self.matrix_high_0 = torch.Tensor(matrix_g_0).cuda()
            self.matrix_high_1 = torch.Tensor(matrix_g_1).cuda()
        else:
            self.matrix_low_0 = torch.Tensor(matrix_h_0)
            self.matrix_low_1 = torch.Tensor(matrix_h_1)
            self.matrix_high_0 = torch.Tensor(matrix_g_0)
            self.matrix_high_1 = torch.Tensor(matrix_g_1)

    def forward(self, input):
        assert len(input.size()) == 4
        self.input_height = input.size()[-2]
        self.input_width = input.size()[-1]
        self.get_matrix()
        return DWTFunction_2D.apply(
            input,
            self.matrix_low_0,
            self.matrix_low_1,
            self.matrix_high_0,
            self.matrix_high_1,
        )


class IDWT_2D(Module):
    def __init__(self, wavename):
        super(IDWT_2D, self).__init__()
        wavelet = pywt.Wavelet(wavename)
        self.band_low = wavelet.dec_lo
        self.band_low.reverse()
        self.band_high = wavelet.dec_hi
        self.band_high.reverse()
        assert len(self.band_low) == len(self.band_high)
        self.band_length = len(self.band_low)
        assert self.band_length % 2 == 0
        self.band_length_half = math.floor(self.band_length / 2)

    def get_matrix(self):
        L1 = np.max((self.input_height, self.input_width))
        L = math.floor(L1 / 2)
        matrix_h = np.zeros((L, L1 + self.band_length - 2))
        matrix_g = np.zeros((L1 - L, L1 + self.band_length - 2))
        end = None if self.band_length_half == 1 else (-self.band_length_half + 1)
        index = 0
        for i in range(L):
            for j in range(self.band_length):
                matrix_h[i, index + j] = self.band_low[j]
            index += 2
        matrix_h_0 = matrix_h[
            0 : (math.floor(self.input_height / 2)),
            0 : (self.input_height + self.band_length - 2),
        ]
        matrix_h_1 = matrix_h[
            0 : (math.floor(self.input_width / 2)),
            0 : (self.input_width + self.band_length - 2),
        ]
        index = 0
        for i in range(L1 - L):
            for j in range(self.band_length):
                matrix_g[i, index + j] = self.band_high[j]
            index += 2
        matrix_g_0 = matrix_g[
            0 : (self.input_height - math.floor(self.input_height / 2)),
            0 : (self.input_height + self.band_length - 2),
        ]
        matrix_g_1 = matrix_g[
            0 : (self.input_width - math.floor(self.input_width / 2)),
            0 : (self.input_width + self.band_length - 2),
        ]
        matrix_h_0 = matrix_h_0[:, (self.band_length_half - 1) : end]
        matrix_h_1 = matrix_h_1[:, (self.band_length_half - 1) : end]
        matrix_h_1 = np.transpose(matrix_h_1)
        matrix_g_0 = matrix_g_0[:, (self.band_length_half - 1) : end]
        matrix_g_1 = matrix_g_1[:, (self.band_length_half - 1) : end]
        matrix_g_1 = np.transpose(matrix_g_1)
        if torch.cuda.is_available():
            self.matrix_low_0 = torch.Tensor(matrix_h_0).cuda()
            self.matrix_low_1 = torch.Tensor(matrix_h_1).cuda()
            self.matrix_high_0 = torch.Tensor(matrix_g_0).cuda()
            self.matrix_high_1 = torch.Tensor(matrix_g_1).cuda()
        else:
            self.matrix_low_0 = torch.Tensor(matrix_h_0)
            self.matrix_low_1 = torch.Tensor(matrix_h_1)
            self.matrix_high_0 = torch.Tensor(matrix_g_0)
            self.matrix_high_1 = torch.Tensor(matrix_g_1)

    def forward(self, LL, LH, HL, HH):
        assert len(LL.size()) == len(LH.size()) == len(HL.size()) == len(HH.size()) == 4
        self.input_height = LL.size()[-2] + HH.size()[-2]
        self.input_width = LL.size()[-1] + HH.size()[-1]
        self.get_matrix()
        return IDWTFunction_2D.apply(
            LL,
            LH,
            HL,
            HH,
            self.matrix_low_0,
            self.matrix_low_1,
            self.matrix_high_0,
            self.matrix_high_1,
        )


class IDWTFunction_2D(Function):
    def forward(
        ctx,
        input_LL,
        input_LH,
        input_HL,
        input_HH,
        matrix_Low_0,
        matrix_Low_1,
        matrix_High_0,
        matrix_High_1,
    ):
        ctx.save_for_backward(matrix_Low_0, matrix_Low_1, matrix_High_0, matrix_High_1)
        L = torch.matmul(input_LH, matrix_High_1.t())
        H = torch.add(
            torch.matmul(input_HL, matrix_Low_1.t()),
            torch.matmul(input_HH, matrix_High_1.t()),
        )
        output = torch.add(
            torch.matmul(matrix_Low_0.t(), L), torch.matmul(matrix_High_0.t(), H)
        )
        return output

    def backward(ctx, grad_output):
        matrix_Low_0, matrix_Low_1, matrix_High_0, matrix_High_1 = ctx.saved_tensors
        grad_L = torch.matmul(matrix_Low_0, grad_output)
        grad_H = torch.matmul(matrix_High_0, grad_output)
        grad_LL = torch.matmul(grad_L, matrix_Low_1)
        grad_LH = torch.matmul(grad_L, matrix_High_1)
        grad_HL = torch.matmul(grad_H, matrix_Low_1)
        grad_HH = torch.matmul(grad_H, matrix_High_1)
        return grad_LL, grad_LH, grad_HL, grad_HH, None, None, None, None


class DWTFunction_2D(Function):
    def forward(ctx, input, matrix_Low_0, matrix_Low_1, matrix_High_0, matrix_High_1):
        ctx.save_for_backward(matrix_Low_0, matrix_Low_1, matrix_High_0, matrix_High_1)
        L = torch.matmul(matrix_Low_0, input)
        H = torch.matmul(matrix_High_0, input)
        LL = torch.matmul(L, matrix_Low_1)
        LH = torch.matmul(L, matrix_High_1)
        HL = torch.matmul(H, matrix_Low_1)
        HH = torch.matmul(H, matrix_High_1)
        return LL, LH, HL, HH

    def backward(ctx, grad_LL, grad_LH, grad_HL, grad_HH):
        matrix_Low_0, matrix_Low_1, matrix_High_0, matrix_High_1 = ctx.saved_tensors
        grad_L = torch.matmul(grad_LH, matrix_High_1.t())
        grad_H = torch.add(
            torch.matmul(grad_HL, matrix_Low_1.t()),
            torch.matmul(grad_HH, matrix_High_1.t()),
        )
        grad_input = torch.add(
            torch.matmul(matrix_Low_0.t(), grad_L),
            torch.matmul(matrix_High_0.t(), grad_H),
        )
        return grad_input, None, None, None, None
