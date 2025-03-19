import torch
import torch.nn as nn
import torch.nn.functional as F

class WaveletUpsample(nn.Module):
    # Wavelet upsample module, replacing traditional upsampling to retain more details
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super(WaveletUpsample, self).__init__()
        self.scale_factor = scale_factor

        # Pre-processing convolution
        self.preconv = nn.Conv2d(in_channels, out_channels*4, 3, 1, 1)

        #TODO Note: In practice, may need to use PyTorch's wavelet library or custom wavelet transform
        # Here we use a simplified version

    def forward(self, x):
        # Feature pre-processing
        x = self.preconv(x)

        batch, channels = x.shape[0], x.shape[1]//4
        h, w = x.shape[2], x.shape[3]

        # Split into low-frequency and high-frequency parts
        ll = x[:, :channels, :, :]
        lh = x[:, channels:channels*2, :, :]
        hl = x[:, channels*2:channels*3, :, :]
        hh = x[:, channels*3:, :, :]

        # Upsample using pixel shuffle (simplified version of wavelet inverse transform)
        y = torch.zeros(batch, channels, h*2, w*2, device=x.device)

        # Fill the upsampled tensor
        y[:, :, 0::2, 0::2] = ll
        y[:, :, 0::2, 1::2] = lh
        y[:, :, 1::2, 0::2] = hl
        y[:, :, 1::2, 1::2] = hh

        return y

class RawWaveletUpsample(nn.Module):
    # Wavelet upsample designed for RAW image processing
    def __init__(self, in_channels, out_channels):
        super(RawWaveletUpsample, self).__init__()

        # Use separate wavelet processing for each color channel
        self.r_branch = WaveletUpsample(in_channels, out_channels//4)
        self.g_branch = WaveletUpsample(in_channels, out_channels//2)
        self.b_branch = WaveletUpsample(in_channels, out_channels//4)

        # Feature fusion
        self.fusion = nn.Conv2d(out_channels, out_channels, 1)

    def forward(self, x):
        # Separate RGGB channels
        # Assuming input is 4-channel representing RGGB
        r = x[:, 0:1, :, :]
        g1 = x[:, 1:2, :, :]
        g2 = x[:, 2:3, :, :]
        b = x[:, 3:4, :, :]

        g = (g1 + g2) / 2

        # Wavelet upsample for each channel
        r_up = self.r_branch(r)
        g_up = self.g_branch(g)
        b_up = self.b_branch(b)

        # Merge three color channels
        combined = torch.cat([r_up, g_up, b_up], dim=1)
        out = self.fusion(combined)

        return out