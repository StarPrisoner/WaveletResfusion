# utils/dwt.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class DWTForward(nn.Module):
    """Optimized Haar wavelet decomposition"""
    def __init__(self, pad_mode='reflect'):
        super().__init__()
        # Define learnable wavelet kernels
        self.ll = nn.Parameter(torch.tensor([[1, 1], [1, 1]], dtype=torch.float32) / 2)
        self.lh = nn.Parameter(torch.tensor([[-1, -1], [1, 1]], dtype=torch.float32) / 2)
        self.hl = nn.Parameter(torch.tensor([[-1, 1], [-1, 1]], dtype=torch.float32) / 2)
        self.hh = nn.Parameter(torch.tensor([[1, -1], [-1, 1]], dtype=torch.float32) / 2)
        self.pad_mode = pad_mode

    def forward(self, x):
        # Automatic padding to ensure decomposability
        B, C, H, W = x.shape
        if H % 2 != 0 or W % 2 != 0:
            x = F.pad(x, (0, 1, 0, 1), mode=self.pad_mode)

        # Separate channel calculation
        ll = F.conv2d(x, self.ll.view(1, 1, 2, 2).expand(C, -1, -1, -1), stride=2, groups=C)
        lh = F.conv2d(x, self.lh.view(1, 1, 2, 2).expand(C, -1, -1, -1), stride=2, groups=C)
        hl = F.conv2d(x, self.hl.view(1, 1, 2, 2).expand(C, -1, -1, -1), stride=2, groups=C)
        hh = F.conv2d(x, self.hh.view(1, 1, 2, 2).expand(C, -1, -1, -1), stride=2, groups=C)

        return ll, (lh, hl, hh)


class DWTInverse(nn.Module):
    """Differentiable wavelet reconstruction"""
    def __init__(self, in_channels):
        super().__init__()
        # Dynamic transposed convolution
        self.conv_t = nn.ConvTranspose2d(
            in_channels * 4, in_channels,
            kernel_size=4,
            stride=2,
            padding=1,
            groups=in_channels
        )
        # Initialize kernel parameters
        kernel = torch.zeros(4, 1, 4, 4)
        kernel[0, 0, 1::2, 1::2] = 1  # LL
        kernel[1, 0, 1::2, ::2] = 1   # LH
        kernel[2, 0, ::2, 1::2] = 1   # HL
        kernel[3, 0, ::2, ::2] = 1    # HH
        self.conv_t.weight.data = kernel.repeat(in_channels, 1, 1, 1) / 2
        self.conv_t.weight.requires_grad = False

    def forward(self, coeffs):
        ll, highs = coeffs
        lh, hl, hh = highs
        # Concatenate along channel dimension
        x = torch.cat([ll, lh, hl, hh], dim=1)
        return self.conv_t(x)