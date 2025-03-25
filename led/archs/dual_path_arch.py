import torch
import torch.nn as nn
import torch.nn.functional as F
from led.utils.registry import ARCH_REGISTRY
import led.utils.noise_map_processor as nmp

from .dual_path_components import (
    DilatedConvChain, HighFrequencyAttention,
    AdaptiveDenoiseGate, ResidualDenoiser,
    DynamicFusion, WaveletUpsample, SharpnessRecovery, DiscreteWaveletUpsample
)

class DualPathBlock(nn.Module):
    """double path block, including detail path and denoising path"""
    def __init__(self, in_channels, out_channels, use_noise_map=False):
        super(DualPathBlock, self).__init__()

        # shared feature extraction
        self.features = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.activation = nn.LeakyReLU(0.2, inplace=True)

        # detail path
        self.detail_path = nn.Sequential(
            DilatedConvChain(out_channels),
            HighFrequencyAttention(out_channels, use_noise_map)
        )

        # denoising path gate mechanism
        self.denoise_gate = AdaptiveDenoiseGate(out_channels, use_noise_map)

        # denoising path residual learning
        self.residual_denoiser = ResidualDenoiser(out_channels)

        # dynamic fusion layer
        self.fusion = DynamicFusion(out_channels, use_noise_map)

        self.use_noise_map = use_noise_map

    def forward(self, x, noise_map=None):
        print(f"\n==== DUALPATHBLOCK FORWARD ====")
        print(f"Block input shape: {x.shape}")
        print(f"Block noise_map: {'None' if noise_map is None else noise_map.shape}")
        print(f"use_noise_map: {self.use_noise_map}")

        # Check for NaN values in the input
        nmp.detect_nan(x, "DualPathBlock input")

        # Check noise map for NaN values if provided
        if noise_map is not None:
            nmp.detect_nan(noise_map, "DualPathBlock noise map")

        # shared feature extraction
        feat = self.features(x)
        feat = self.activation(feat)
        print(f"Feature extraction output shape: {feat.shape}")

        #----------------------------- DETAIL PATH -----------------------------#
        print("Detail path processing...")
        if isinstance(self.detail_path[-1], HighFrequencyAttention) and self.use_noise_map and noise_map is not None:
            detail = self.detail_path[0](feat)
            print(f"Detail path intermediate shape: {detail.shape}")
            detail = nmp.apply_to_module(
            detail, self.detail_path[-1], noise_map, detail.size(1)
        )
        else:
            detail = self.detail_path(feat)
        nmp.detect_nan(detail, "detail path output")
        print(f"Detail path output shape: {detail.shape}")
        #--------------------------- DENOISING PATH ---------------------------#
        print("Denoising path processing...")
        # Compute adaptive gating mechanism
        if self.use_noise_map and noise_map is not None:
            # Get denoising gate with adjusted noise map
            gate = nmp.apply_to_module(
                feat, self.denoise_gate, noise_map, feat.size(1)
            )
        else:
            # Get denoising gate without noise map
            gate = self.denoise_gate(feat)

        print(f"Gate shape: {gate.shape}")
        # Apply residual denoising with gating
        denoise = self.residual_denoiser(feat, gate)
        nmp.detect_nan(denoise, "denoising path output")
        print(f"Denoising path output shape: {denoise.shape}")
        #--------------------------- DYNAMIC FUSION ---------------------------#
        print("Fusion processing...")
         # Fuse outputs from both paths using content-aware mechanism
        if self.use_noise_map and noise_map is not None:
            # wrap the fusion call with correct parameters
            fusion_func = lambda d, n: self.fusion(d, denoise, feat, n)
            out = nmp.apply_to_module(
                detail, fusion_func, noise_map, feat.size(1)
            )
        else:
            out = self.fusion(detail, denoise, feat)

        nmp.detect_nan(out, "fusion output")
        print(f"Fusion output shape: {out.shape}")
        return out

@ARCH_REGISTRY.register()
class DualPathUNet(nn.Module):
    """double path U-Net, apply double path design on each scale of U-Net"""
    def __init__(self, in_channels=4, out_channels=4, base_channels=64,
                 dilated_rates=None, use_wavelet_upsample=True,
                 use_sharpness_recovery=True, use_noise_map=False):
        super(DualPathUNet, self).__init__()

        self.base_channels = base_channels
        self.use_noise_map = use_noise_map
        self.use_wavelet_upsample = use_wavelet_upsample
        self.use_sharpness_recovery = use_sharpness_recovery

        # enc1_in_channels = in_channels * 2 if use_noise_map else in_channels
        enc1_in_channels = in_channels

        # encoder
        self.enc1 = DualPathBlock(enc1_in_channels, base_channels, use_noise_map)
        self.enc2 = DualPathBlock(base_channels, base_channels*2, use_noise_map)
        self.enc3 = DualPathBlock(base_channels*2, base_channels*4, use_noise_map)

        # bottleneck
        self.bottleneck = DualPathBlock(base_channels*4, base_channels*8, use_noise_map)

        # decoder with skip connections
        self.dec3 = DualPathBlock(base_channels*4+base_channels*4, base_channels*4, use_noise_map)
        self.dec2 = DualPathBlock(base_channels*2+base_channels*2, base_channels*2, use_noise_map)
        self.dec1 = DualPathBlock(base_channels+base_channels, base_channels, use_noise_map)

        # downsample and upsample
        self.down = nn.MaxPool2d(2)
        self.up3 = nn.ConvTranspose2d(base_channels*8, base_channels*4, 2, stride=2)
        self.up2 = nn.ConvTranspose2d(base_channels*4, base_channels*2, 2, stride=2)

        # final upsample, with optional wavelet upsample
        if use_wavelet_upsample:
            self.up1 = DiscreteWaveletUpsample(base_channels*2, base_channels)
        else:
            self.up1 = nn.ConvTranspose2d(base_channels*2, base_channels, 2, stride=2)

        # output layer
        self.final = nn.Conv2d(base_channels, out_channels, 1)

        # sharpness recovery
        if use_sharpness_recovery:
            self.sharpness_recovery = SharpnessRecovery(out_channels, use_noise_map)

    def forward(self, x, noise_map=None):
        print("\n==== DUALPATHNET FORWARD ====")
        print(f"Input x shape: {x.shape}")
        print(f"Input noise_map: {'None' if noise_map is None else noise_map.shape}")
        print(f"self.use_noise_map: {self.use_noise_map}")

        nmp.detect_nan(x, "input image")

        if noise_map is not None:
            nmp.detect_nan(noise_map, "input noise map")

        # concatenate noise map if needed
        if self.use_noise_map and noise_map is not None:
            # noise map for different size
            print("Creating multiscale noise maps")
            noise_maps = nmp.create_multiscale_maps(
                noise_map, scales=[1, 2, 4, 8]
            )
            noise_maps = {
                'original': noise_maps['scale_1'],
                'down1': noise_maps['scale_2'],
                'down2': noise_maps['scale_4'],
                'down3': noise_maps['scale_8']
            }
            print(f"Created {len(noise_maps)} noise maps")
            for k, v in noise_maps.items():
                if v is not None:
                    print(f"  {k}: {v.shape}")
            # x_input = torch.cat([x, noise_map], dim=1)
            x_input = x
        else:
            print("Not using noise maps or noise_map is None")
            noise_maps = {k: None for k in ['original', 'down1', 'down2', 'down3']}
            x_input = x

        print(f"x_input shape: {x_input.shape}")

        #--------------------------- ENCODER PATH ---------------------------#

        # [1, 4/8, 1024, 1024] -> [1, 64, 1024, 1024]
        enc1 = nmp.apply_to_module(
            x_input, self.enc1, noise_maps['original'], self.base_channels
        )
        # [1, 64, 1024, 1024] -> [1, 128, 512, 512]
        enc1_down = self.down(enc1)
        enc2 = nmp.apply_to_module(
            enc1_down, self.enc2, noise_maps['down1'], self.base_channels * 2
        )
        # [1, 128, 512, 512] -> [1, 256, 256, 256]
        enc2_down = self.down(enc2)
        enc3 = nmp.apply_to_module(
            enc2_down, self.enc3, noise_maps['down2'], self.base_channels * 4
        )

        #--------------------------- BOTTLENECK ---------------------------#

        # [1, 256, 256, 256] -> [1, 512, 128, 128]
        enc3_down = self.down(enc3)
        bottleneck = nmp.apply_to_module(
            enc3_down, self.bottleneck
        )
        nmp.detect_nan(bottleneck, "bottleneck")

        #--------------------------- DECODER PATH ---------------------------#

        # [1, 512, 128, 128] -> [[1, 256, 256, 256]] + [1, 256, 256, 256] -> [1, 256, 256, 256]
        bottleneck_up = self.up3(bottleneck)
        dec3_input = torch.cat([bottleneck_up, enc3], dim=1)
        dec3 = nmp.apply_to_module(
            dec3_input, self.dec3, noise_maps['down2'], self.base_channels * 4
        )
        # [1, 256, 256, 256] -> [1, 128, 512, 512]
        dec3_up = self.up2(dec3)
        dec2_input = torch.cat([dec3_up, enc2], dim=1)
        dec2 = nmp.apply_to_module(
            dec2_input, self.dec2, noise_maps['down1'], self.base_channels * 2
        )
        # [1, 128, 512, 512] -> [1, 64, 1024, 1024]
        dec2_up = self.up1(dec2)
        dec1_input = torch.cat([dec2_up, enc1], dim=1)
        dec1 = nmp.apply_to_module(
            dec1_input, self.dec1, noise_maps['original'], self.base_channels
        )

        #--------------------------- FINAL OUTPUT ---------------------------#

        out = self.final(dec1)
        nmp.detect_nan(out, "final network output")

        # sharpness recovery
        if self.use_sharpness_recovery:
            out = nmp.apply_to_module(
                out, self.sharpness_recovery, noise_maps['original'], out.size(1)
            )

        return out