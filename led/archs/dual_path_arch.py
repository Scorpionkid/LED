import torch
import torch.nn as nn
import torch.nn.functional as F
from led.utils.registry import ARCH_REGISTRY

from .dual_path_components import (
    DilatedConvChain, HighFrequencyAttention,
    AdaptiveDenoiseGate, ResidualDenoiser,
    DynamicFusion, WaveletUpsample, SharpnessRecovery
)

class DualPathBlock(nn.Module):
    """double path block, including detail path and denoising path"""
    def __init__(self, in_channels, out_channels):
        super(DualPathBlock, self).__init__()

        # shared feature extraction
        self.features = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.activation = nn.LeakyReLU(0.2, inplace=True)

        # detail path
        self.detail_path = nn.Sequential(
            DilatedConvChain(out_channels),
            HighFrequencyAttention(out_channels)
        )

        # denoising path gate mechanism
        self.denoise_gate = AdaptiveDenoiseGate(out_channels)

        # denoising path residual learning
        self.residual_denoiser = ResidualDenoiser(out_channels)

        # dynamic fusion layer
        self.fusion = DynamicFusion(out_channels)

    def forward(self, x):
        # shared feature extraction
        feat = self.features(x)
        feat = self.activation(feat)

        # detail path processing
        detail = self.detail_path(feat)

        # denoising path processing
        gate = self.denoise_gate(feat)
        denoise = self.residual_denoiser(feat, gate)

        # dynamic fusion
        out = self.fusion(detail, denoise, feat)

        return out

@ARCH_REGISTRY.register()
class DualPathUNet(nn.Module):
    """double path U-Net, apply double path design on each scale of U-Net"""
    def __init__(self, in_channels=4, out_channels=4, base_channels=64,
                 dilated_rates=None, use_wavelet_upsample=True,
                 use_sharpness_recovery=True):
        super(DualPathUNet, self).__init__()

        self.use_wavelet_upsample = use_wavelet_upsample
        self.use_sharpness_recovery = use_sharpness_recovery

        # encoder
        self.enc1 = DualPathBlock(in_channels, base_channels)
        self.enc2 = DualPathBlock(base_channels, base_channels*2)
        self.enc3 = DualPathBlock(base_channels*2, base_channels*4)

        # bottleneck
        self.bottleneck = DualPathBlock(base_channels*4, base_channels*8)

        # decoder with skip connections
        self.dec3 = DualPathBlock(base_channels*8+base_channels*4, base_channels*4)
        self.dec2 = DualPathBlock(base_channels*4+base_channels*2, base_channels*2)
        self.dec1 = DualPathBlock(base_channels*2+base_channels, base_channels)

        # downsample and upsample
        self.down = nn.MaxPool2d(2)
        self.up3 = nn.ConvTranspose2d(base_channels*8, base_channels*4, 2, stride=2)
        self.up2 = nn.ConvTranspose2d(base_channels*4, base_channels*2, 2, stride=2)

        # final upsample, with optional wavelet upsample
        if use_wavelet_upsample:
            self.up1 = WaveletUpsample(base_channels*2, base_channels)
        else:
            self.up1 = nn.ConvTranspose2d(base_channels*2, base_channels, 2, stride=2)

        # output layer
        self.final = nn.Conv2d(base_channels, out_channels, 1)

        # sharpness recovery
        if use_sharpness_recovery:
            self.sharpness_recovery = SharpnessRecovery(out_channels)

    def forward(self, x):
        # encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.down(enc1))
        enc3 = self.enc3(self.down(enc2))

        # bottleneck
        bottleneck = self.bottleneck(self.down(enc3))

        # decoder
        dec3 = self.dec3(torch.cat([bottleneck, enc3], dim=1))
        dec2 = self.dec2(torch.cat([dec3, enc2], dim=1))
        dec1 = self.dec1(torch.cat([dec2, enc1], dim=1))

        # final output
        out = self.final(dec1)

        # sharpness recovery
        if self.use_sharpness_recovery:
            out = self.sharpness_recovery(out)

        return out
