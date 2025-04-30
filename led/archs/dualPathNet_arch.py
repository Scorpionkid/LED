import torch
import torch.nn as nn
import torch.nn.functional as F
from led.utils.registry import ARCH_REGISTRY
import led.utils.noise_map_processor as nmp

from .dual_path_components import (
    DynamicFusion, AGF,
    WaveletUpsample, DiscreteWaveletUpsample,
    SharpnessRecovery,
    RAWTextureDetector,
    EnhancedDenoisePath,
    EnhancedDetailPath, EDP2
)

class DualPathBlock(nn.Module):
    """双路径块，包含细节路径和降噪路径"""
    def __init__(self, in_channels, out_channels, texture_params=None, heads=1):
        super(DualPathBlock, self).__init__()

        # 特征提取
        self.features = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.activation = nn.LeakyReLU(0.2, inplace=True)

        # 纹理参数
        texture_params = texture_params or {}

        # 双路径组件
        self.detail_path = EDP2(out_channels, num_heads=heads)
        self.denoise_path = EnhancedDenoisePath(out_channels, num_heads=heads)
        self.fusion = AGF(out_channels, use_noise_map=True, use_texture_mask=True)

    def forward(self, x, noise_map=None, texture_mask=None):
        # 特征提取
        feat = self.features(x)
        feat = self.activation(feat)

        # 细节路径
        detail = self.detail_path(feat, noise_map, texture_mask)

        # 降噪路径
        denoise = self.denoise_path(feat, noise_map, texture_mask)

        # 动态融合
        output = self.fusion(detail, denoise, feat, noise_map, texture_mask)

        return output

@ARCH_REGISTRY.register()
class DualPathUNet_E1_SMP(nn.Module):
    def __init__(self, in_channels=4, out_channels=4, base_channels=64, heads=[1,2,4,8],
                 dilated_rates=None, use_wavelet_upsample=True,
                 use_sharpness_recovery=True, use_noise_map=True,
                 use_texture_detection=True, texture_params=None,
                 **kwargs):
        super(DualPathUNet_E1_SMP, self).__init__()

        self.base_channels = base_channels
        self.use_wavelet_upsample = use_wavelet_upsample
        self.use_sharpness_recovery = use_sharpness_recovery
        self.use_noise_map = use_noise_map
        self.use_texture_detection = use_texture_detection

        # 纹理参数
        self.texture_params = {
            'texture_gate': 0.5,
            'texture_suppress_factor': 0.7,
            'fusion_texture_boost': 0.5,
            'sharpness_texture_boost': 0.3,
        }
        if texture_params:
            self.texture_params.update(texture_params)

        # 纹理检测器
        if use_texture_detection:
            texture_detector_params = self.texture_params.get('texture_detector_params', {})
            window_sizes = texture_detector_params.get('window_sizes', [5, 9, 17])
            adaptive_thresh = texture_detector_params.get('adaptive_thresh', True)
            noise_sensitivity = texture_detector_params.get('noise_sensitivity', 3.0)

            self.texture_detector = RAWTextureDetector(
                window_sizes=window_sizes,
                adaptive_thresh=adaptive_thresh,
                raw_channels=in_channels,
                noise_sensitivity=noise_sensitivity
            )

        # 编码器路径
        self.enc1 = DualPathBlock(in_channels, base_channels, self.texture_params, heads[0])
        self.enc2 = DualPathBlock(base_channels, base_channels*2, self.texture_params, heads[1])
        self.enc3 = DualPathBlock(base_channels*2, base_channels*4, self.texture_params, heads[2])

        # 瓶颈
        self.bottleneck = DualPathBlock(base_channels*4, base_channels*8, self.texture_params, heads[3])

        # 解码器路径
        self.dec3 = DualPathBlock(base_channels*4+base_channels*4, base_channels*4, self.texture_params, heads[2])
        self.dec2 = DualPathBlock(base_channels*2+base_channels*2, base_channels*2, self.texture_params, heads[1])
        self.dec1 = DualPathBlock(base_channels+base_channels, base_channels, self.texture_params, heads[0])

        # 下采样和上采样
        self.down = nn.MaxPool2d(2)
        self.up3 = nn.ConvTranspose2d(base_channels*8, base_channels*4, 2, stride=2)
        self.up2 = nn.ConvTranspose2d(base_channels*4, base_channels*2, 2, stride=2)

        # 小波上采样
        if use_wavelet_upsample:
            self.up1 = DiscreteWaveletUpsample(base_channels*2, base_channels)
        else:
            self.up1 = nn.ConvTranspose2d(base_channels*2, base_channels, 2, stride=2)

        # 输出层
        self.final = nn.Conv2d(base_channels, out_channels, 1)

        # 锐度恢复
        if use_sharpness_recovery:
            self.sharpness_recovery = SharpnessRecovery(
                out_channels,
                use_noise_map=True,
                use_texture_mask=True,
                sharpness_texture_boost=self.texture_params.get('sharpness_texture_boost', 0.3)
            )

    def forward(self, x, noise_map=None, texture_mask=None):

        if self.use_texture_detection:
            computed_texture_mask = self.texture_detector(x, noise_map)
            texture_mask = computed_texture_mask if texture_mask is None else texture_mask
            texture_mask = nmp.standardize_map(texture_mask)

            texture_maps = nmp.create_multiscale_maps(texture_mask, scales=[1, 2, 4, 6])
            tm_original = texture_maps['scale_1']
            tm_down1 = texture_maps['scale_2']
            tm_down2 = texture_maps['scale_4']
            tm_down3 = texture_maps['scale_6']
        else:
            computed_texture_mask = None
            tm_original = tm_down1 = tm_down2 = tm_down3 = None

        if self.use_noise_map and noise_map is not None:
            noise_map = nmp.standardize_map(noise_map)

            noise_maps = nmp.create_multiscale_maps(noise_map, scales=[1, 2, 4, 6])
            nm_original = noise_maps['scale_1']
            nm_down1 = noise_maps['scale_2']
            nm_down2 = noise_maps['scale_4']
            nm_down3 = noise_maps['scale_6']
        else:
            nm_original = nm_down1 = nm_down2 = nm_down3 = None

        enc1 = self.enc1(x, nm_original, tm_original)
        enc1_down = self.down(enc1)

        enc2 = self.enc2(enc1_down, nm_down1, tm_down1)
        enc2_down = self.down(enc2)

        enc3 = self.enc3(enc2_down, nm_down2, tm_down2)
        enc3_down = self.down(enc3)

        bottleneck = self.bottleneck(enc3_down, nm_down3, tm_down3)

        bottleneck_up = self.up3(bottleneck)
        dec3_input = torch.cat([bottleneck_up, enc3], dim=1)
        dec3 = self.dec3(dec3_input, nm_down2, tm_down2)

        dec3_up = self.up2(dec3)
        dec2_input = torch.cat([dec3_up, enc2], dim=1)
        dec2 = self.dec2(dec2_input, nm_down1, tm_down1)

        dec2_up = self.up1(dec2)
        dec1_input = torch.cat([dec2_up, enc1], dim=1)
        dec1 = self.dec1(dec1_input, nm_original, tm_original)

        # 输出层
        out = self.final(dec1)

        if self.use_sharpness_recovery:
            out = self.sharpness_recovery(out, nm_original, tm_original)

        return out, computed_texture_mask