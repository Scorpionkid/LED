import torch
import torch.nn as nn
import torch.nn.functional as F
from led.utils.registry import ARCH_REGISTRY

from .dual_path_restormer_components import EnhancedDetailPath, EnhancedDenoisePath, EnhancedFusion
from .dual_path_components import SharpnessRecovery, DiscreteWaveletUpsample, RAWTextureDetector
import led.utils.noise_map_processor as nmp

class DualPathRestormerBlock(nn.Module):
    """双路径Restormer块

    结合增强的细节路径和降噪路径
    """

    def __init__(self, in_channels, out_channels, num_heads=1, use_noise_map=False,
                use_texture_in_detail=False,
                use_texture_in_denoise=False,
                use_texture_in_fusion=False,
                texture_params=None):
        """初始化

        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
            num_heads: 注意力头数
            use_noise_map: 是否使用噪声图
            use_texture_in_detail: 细节路径是否使用纹理掩码
            use_texture_in_denoise: 降噪路径是否使用纹理掩码
            use_texture_in_fusion: 融合层是否使用纹理掩码
            texture_params: 纹理相关参数
        """
        super(DualPathRestormerBlock, self).__init__()

        # 特征提取
        self.features = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.activation = nn.LeakyReLU(0.2, inplace=True)

        # 纹理参数
        texture_params = texture_params or {}
        texture_gate = texture_params.get('texture_gate', 0.5)
        texture_suppress_factor = texture_params.get('texture_suppress_factor', 0.7)
        fusion_texture_boost = texture_params.get('fusion_texture_boost', 0.5)

        # 细节路径
        self.detail_path = EnhancedDetailPath(
            out_channels,
            num_heads=num_heads,
            use_noise_map=use_noise_map and use_texture_in_detail
        )

        # 降噪路径
        self.denoise_path = EnhancedDenoisePath(
            out_channels,
            use_noise_map=use_noise_map and use_texture_in_denoise,
            texture_suppress_factor=texture_suppress_factor
        )

        # 融合层
        self.fusion = EnhancedFusion(
            out_channels,
            use_noise_map=use_noise_map and use_texture_in_fusion,
            use_texture_mask=use_texture_in_fusion,
            fusion_texture_boost=fusion_texture_boost
        )

        # 设置标志
        self.use_noise_map = use_noise_map
        self.use_texture_in_detail = use_texture_in_detail
        self.use_texture_in_denoise = use_texture_in_denoise
        self.use_texture_in_fusion = use_texture_in_fusion

    def forward(self, x, noise_map=None, texture_mask=None):
        """前向传播

        Args:
            x: 输入特征 [B, C, H, W]
            noise_map: 可选的噪声图 [B, 1, H, W]
            texture_mask: 可选的纹理掩码 [B, 1, H, W]

        Returns:
            处理后的特征 [B, C, H, W]
        """
        # 检查输入是否有NaN
        nmp.detect_nan(x, "DualPathRestormerBlock input")

        # 特征提取
        feat = self.features(x)
        feat = self.activation(feat)

        # 细节路径
        detail_input = feat
        detail_noise = noise_map if self.use_noise_map and self.use_texture_in_detail else None
        detail_texture = texture_mask if self.use_texture_in_detail else None
        detail = self.detail_path(detail_input, detail_noise)

        # 降噪路径
        denoise_input = feat
        denoise_noise = noise_map if self.use_noise_map and self.use_texture_in_denoise else None
        denoise_texture = texture_mask if self.use_texture_in_denoise else None
        denoise = self.denoise_path(denoise_input, denoise_noise, denoise_texture)

        # 融合
        fusion_noise = noise_map if self.use_noise_map and self.use_texture_in_fusion else None
        fusion_texture = texture_mask if self.use_texture_in_fusion else None
        output = self.fusion(detail, denoise, feat, fusion_noise, fusion_texture)

        return output

@ARCH_REGISTRY.register()
class DualPathRestormer(nn.Module):
    """双路径Restormer架构

    结合双路径设计和Restormer组件的图像恢复网络
    """

    def __init__(self, in_channels=4, out_channels=4, base_channels=64,
                 dilated_rates=None, use_wavelet_upsample=True,
                 use_sharpness_recovery=True, use_noise_map=False,
                 use_texture_detection=False,
                 use_texture_in_detail=None,
                 use_texture_in_denoise=None,
                 use_texture_in_fusion=None,
                 use_texture_in_recovery=None,
                 texture_params=None):
        """初始化

        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
            base_channels: 基础通道数
            dilated_rates: 空洞卷积膨胀率
            use_wavelet_upsample: 是否使用小波上采样
            use_sharpness_recovery: 是否使用锐度恢复
            use_noise_map: 是否使用噪声图
            use_texture_detection: 是否使用纹理检测
            use_texture_in_detail: 细节路径是否使用纹理
            use_texture_in_denoise: 降噪路径是否使用纹理
            use_texture_in_fusion: 融合是否使用纹理
            use_texture_in_recovery: 锐度恢复是否使用纹理
            texture_params: 纹理相关参数
        """
        super(DualPathRestormer, self).__init__()

        self.base_channels = base_channels
        self.use_noise_map = use_noise_map
        self.use_wavelet_upsample = use_wavelet_upsample
        self.use_sharpness_recovery = use_sharpness_recovery

        # 纹理检测相关
        self.use_texture_detection = use_texture_detection
        self.use_texture_in_detail = use_texture_in_detail if use_texture_in_detail is not None else use_texture_detection
        self.use_texture_in_denoise = use_texture_in_denoise if use_texture_in_denoise is not None else use_texture_detection
        self.use_texture_in_fusion = use_texture_in_fusion if use_texture_in_fusion is not None else use_texture_detection
        self.use_texture_in_recovery = use_texture_in_recovery if use_texture_in_recovery is not None else use_texture_detection

        # 输入投影
        self.conv_first = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1, bias=False)

        # 纹理检测器
        if use_texture_detection:
            self.texture_params = {
                'texture_gate': 0.5,
                'texture_suppress_factor': 0.7,
                'fusion_texture_boost': 0.5,
                'sharpness_texture_boost': 0.3,
            }
            if texture_params is not None:
                self.texture_params.update(texture_params)

            # 纹理检测器参数
            texture_detector_params = texture_params.get('texture_detector_params', {})

            window_sizes = texture_detector_params.get('window_sizes', [5, 9, 17])
            adaptive_thresh = texture_detector_params.get('adaptive_thresh', True)
            noise_sensitivity = texture_detector_params.get('noise_sensitivity', 3.0)

            self.texture_detector = RAWTextureDetector(
                window_sizes=window_sizes,
                adaptive_thresh=adaptive_thresh,
                raw_channels=in_channels,
                noise_sensitivity=noise_sensitivity
            )

        # 编码器
        self.encoders = nn.ModuleList([
            DualPathRestormerBlock(base_channels, base_channels, num_heads=1,
                                 use_noise_map=use_noise_map,
                                 use_texture_in_detail=self.use_texture_in_detail,
                                 use_texture_in_denoise=self.use_texture_in_denoise,
                                 use_texture_in_fusion=self.use_texture_in_fusion,
                                 texture_params=self.texture_params),
            DualPathRestormerBlock(base_channels, base_channels*2, num_heads=2,
                                 use_noise_map=use_noise_map,
                                 use_texture_in_detail=self.use_texture_in_detail,
                                 use_texture_in_denoise=self.use_texture_in_denoise,
                                 use_texture_in_fusion=self.use_texture_in_fusion,
                                 texture_params=self.texture_params),
            DualPathRestormerBlock(base_channels*2, base_channels*4, num_heads=4,
                                 use_noise_map=use_noise_map,
                                 use_texture_in_detail=self.use_texture_in_detail,
                                 use_texture_in_denoise=self.use_texture_in_denoise,
                                 use_texture_in_fusion=self.use_texture_in_fusion,
                                 texture_params=self.texture_params)
        ])

        # 瓶颈层
        self.bottleneck = DualPathRestormerBlock(base_channels*4, base_channels*8, num_heads=8,
                                              use_noise_map=use_noise_map,
                                              use_texture_in_detail=self.use_texture_in_detail,
                                              use_texture_in_denoise=self.use_texture_in_denoise,
                                              use_texture_in_fusion=self.use_texture_in_fusion,
                                              texture_params=self.texture_params)

        # 解码器
        self.decoders = nn.ModuleList([
            DualPathRestormerBlock(base_channels*8, base_channels*4, num_heads=4,
                                 use_noise_map=use_noise_map,
                                 use_texture_in_detail=self.use_texture_in_detail,
                                 use_texture_in_denoise=self.use_texture_in_denoise,
                                 use_texture_in_fusion=self.use_texture_in_fusion,
                                 texture_params=self.texture_params),
            DualPathRestormerBlock(base_channels*4, base_channels*2, num_heads=2,
                                 use_noise_map=use_noise_map,
                                 use_texture_in_detail=self.use_texture_in_detail,
                                 use_texture_in_denoise=self.use_texture_in_denoise,
                                 use_texture_in_fusion=self.use_texture_in_fusion,
                                 texture_params=self.texture_params),
            DualPathRestormerBlock(base_channels*2, base_channels, num_heads=1,
                                 use_noise_map=use_noise_map,
                                 use_texture_in_detail=self.use_texture_in_detail,
                                 use_texture_in_denoise=self.use_texture_in_denoise,
                                 use_texture_in_fusion=self.use_texture_in_fusion,
                                 texture_params=self.texture_params)
        ])

        # 下采样
        self.down = nn.MaxPool2d(2)

        # 上采样
        self.ups = nn.ModuleList([
            nn.ConvTranspose2d(base_channels*8, base_channels*4, 2, stride=2),
            nn.ConvTranspose2d(base_channels*4, base_channels*2, 2, stride=2),
        ])

        # 最终上采样，可选小波上采样
        if use_wavelet_upsample:
            self.up_final = DiscreteWaveletUpsample(base_channels*2, base_channels)
        else:
            self.up_final = nn.ConvTranspose2d(base_channels*2, base_channels, 2, stride=2)

        # 精炼阶段
        self.refine = DualPathRestormerBlock(base_channels, base_channels, num_heads=1,
                                           use_noise_map=use_noise_map,
                                           use_texture_in_detail=self.use_texture_in_detail,
                                           use_texture_in_denoise=self.use_texture_in_denoise,
                                           use_texture_in_fusion=self.use_texture_in_fusion,
                                           texture_params=self.texture_params)

        # 输出层
        self.conv_last = nn.Conv2d(base_channels, out_channels, 1, bias=False)

        # 锐度恢复
        if use_sharpness_recovery:
            self.sharpness_recovery = SharpnessRecovery(
                out_channels,
                use_noise_map=use_noise_map,
                use_texture_mask=self.use_texture_in_recovery,
                sharpness_texture_boost=texture_params.get('sharpness_texture_boost', 0.3) if texture_params else 0.3
            )

    def forward(self, x, noise_map=None, texture_mask=None):
        """前向传播

        Args:
            x: 输入图像 [B, in_channels, H, W]
            noise_map: 可选的噪声图 [B, 1, H, W]
            texture_mask: 可选的纹理掩码 [B, 1, H, W]

        Returns:
            恢复的图像 [B, out_channels, H, W]
            纹理掩码 [B, 1, H, W] 或 None
        """
        # 检查输入
        nmp.detect_nan(x, "input image")

        if noise_map is not None:
            nmp.detect_nan(noise_map, "input noise map")

        # 纹理检测
        computed_texture_mask = None
        if self.use_texture_detection:
            computed_texture_mask = self.texture_detector(x, noise_map)
            if texture_mask is None:
                texture_mask = computed_texture_mask

            nmp.detect_nan(texture_mask, "texture mask")

            # 创建多尺度纹理掩码
            texture_maps = nmp.create_multiscale_maps(
                texture_mask, scales=[1, 2, 4, 8]
            )

        # 创建多尺度噪声图
        if self.use_noise_map and noise_map is not None:
            noise_maps = nmp.create_multiscale_maps(
                noise_map, scales=[1, 2, 4, 8]
            )
        else:
            noise_maps = {k: None for k in [f'scale_{i}' for i in [1, 2, 4, 8]]}

        # 初始特征提取
        feat = self.conv_first(x)

        # 编码器阶段
        encoder_feats = []

        # 第一层编码器
        enc1 = self.encoders[0](feat, noise_maps['scale_1'],
                               texture_maps['scale_1'] if self.use_texture_detection else None)
        encoder_feats.append(enc1)

        # 第二层编码器
        enc1_down = self.down(enc1)
        enc2 = self.encoders[1](enc1_down, noise_maps['scale_2'],
                               texture_maps['scale_2'] if self.use_texture_detection else None)
        encoder_feats.append(enc2)

        # 第三层编码器
        enc2_down = self.down(enc2)
        enc3 = self.encoders[2](enc2_down, noise_maps['scale_4'],
                               texture_maps['scale_4'] if self.use_texture_detection else None)
        encoder_feats.append(enc3)

        # 瓶颈层
        enc3_down = self.down(enc3)
        bottleneck = self.bottleneck(enc3_down, noise_maps['scale_8'],
                                   texture_maps['scale_8'] if self.use_texture_detection else None)

        # 解码器阶段
        # 第一层解码器
        bottleneck_up = self.ups[0](bottleneck)
        dec3_cat = torch.cat([bottleneck_up, encoder_feats[-1]], dim=1)
        dec3 = self.decoders[0](dec3_cat, noise_maps['scale_4'],
                               texture_maps['scale_4'] if self.use_texture_detection else None)

        # 第二层解码器
        dec3_up = self.ups[1](dec3)
        dec2_cat = torch.cat([dec3_up, encoder_feats[-2]], dim=1)
        dec2 = self.decoders[1](dec2_cat, noise_maps['scale_2'],
                               texture_maps['scale_2'] if self.use_texture_detection else None)

        # 第三层解码器
        dec2_up = self.up_final(dec2)
        dec1_cat = torch.cat([dec2_up, encoder_feats[-3]], dim=1)
        dec1 = self.decoders[2](dec1_cat, noise_maps['scale_1'],
                               texture_maps['scale_1'] if self.use_texture_detection else None)

        # 精炼阶段
        refined = self.refine(dec1, noise_maps['scale_1'],
                            texture_maps['scale_1'] if self.use_texture_detection else None)

        # 输出
        out = self.conv_last(refined)

        # 锐度恢复
        if self.use_sharpness_recovery:
            if self.use_texture_in_recovery and computed_texture_mask is not None:
                out = self.sharpness_recovery(out, noise_maps['scale_1'], texture_maps['scale_1'])
            else:
                out = self.sharpness_recovery(out, noise_maps['scale_1'], None)

        return out, computed_texture_mask