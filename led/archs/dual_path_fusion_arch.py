import torch
import torch.nn as nn
import torch.nn.functional as F
from led.utils.registry import ARCH_REGISTRY
import led.utils.noise_map_processor as nmp

from .dual_path_components import (
    DynamicFusion, AGF,
    WaveletUpsample, SharpnessRecovery, DiscreteWaveletUpsample,
    RAWTextureDetector,
    EnhancedDenoisePath, EnhancedDetailPath
)

# 新增：独立路径编码器模块
class IndependentPathEncoder(nn.Module):
    """编码器模块，独立处理细节或降噪路径，不进行融合"""
    def __init__(self, in_channels, out_channels, is_detail_path=True,
                 use_noise_map=False, use_texture_detection=False,
                 heads=1, texture_params=None):
        super(IndependentPathEncoder, self).__init__()

        # 特征提取
        self.features = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.activation = nn.LeakyReLU(0.2, inplace=True)

        # 根据路径类型选择相应模块
        self.is_detail_path = is_detail_path
        if is_detail_path:
            self.path = EnhancedDetailPath(out_channels, heads, use_noise_map)
        else:
            self.path = EnhancedDenoisePath(out_channels, heads)

        self.use_noise_map = use_noise_map
        self.use_texture_detection = use_texture_detection

    def forward(self, x, noise_map=None, texture_mask=None):
        # 特征提取
        feat = self.features(x)
        feat = self.activation(feat)

        # 路径处理
        if self.is_detail_path:
            if self.use_texture_detection and texture_mask is not None:
                if self.use_noise_map and noise_map is not None:
                    output = nmp.apply_to_module(
                        feat, self.path, noise_map, texture_mask
                    )
                else:
                    output = nmp.apply_to_module(
                        feat, self.path, texture_mask=texture_mask
                    )
            else:
                if self.use_noise_map and noise_map is not None:
                    output = nmp.apply_to_module(
                        feat, self.path, noise_map
                    )
                else:
                    output = self.path(feat)
        else:
            # 降噪路径处理
            if self.use_texture_detection and texture_mask is not None:
                if self.use_noise_map and noise_map is not None:
                    output = nmp.apply_to_module(
                        feat, self.path, noise_map, texture_mask
                    )
                else:
                    output = nmp.apply_to_module(
                        feat, self.path, texture_mask=texture_mask
                    )
            else:
                if self.use_noise_map and noise_map is not None:
                    output = nmp.apply_to_module(
                        feat, self.path, noise_map
                    )
                else:
                    output = self.path(feat)

        return output

# 修改：原始的DualPathBlock保留但主要用于解码器阶段
class DualPathBlock(nn.Module):
    """double path block, including detail path and denoising path"""
    def __init__(self, in_channels, out_channels, use_noise_map=False,
                use_texture_in_detail=False,
                use_texture_in_denoise=False,
                use_texture_in_fusion=False,
                texture_params=None, heads=1):
        super(DualPathBlock, self).__init__()

        # shared feature extraction
        self.features = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.activation = nn.LeakyReLU(0.2, inplace=True)

        texture_params = texture_params or {}
        texture_gate = texture_params.get('texture_gate', 0.3)
        texture_suppress_factor = texture_params.get('texture_suppress_factor', 0.7)
        fusion_texture_boost = texture_params.get('fusion_texture_boost', 0.5)

        # detail path
        num_heads = heads
        self.detail_path = EnhancedDetailPath(out_channels, num_heads, use_noise_map)

        self.denoise_path = EnhancedDenoisePath(out_channels, num_heads)

        # dynamic fusion layer
        self.fusion = AGF(out_channels, num_heads, use_noise_map, use_texture_in_fusion)
        # self.fusion = DynamicFusion(out_channels, use_noise_map, use_texture_in_fusion, fusion_texture_boost=fusion_texture_boost)

        self.use_noise_map = use_noise_map
        self.use_texture_in_detail = use_texture_in_detail
        self.use_texture_in_denoise = use_texture_in_denoise
        self.use_texture_in_fusion = use_texture_in_fusion

    def forward(self, x, noise_map=None, texture_mask=None):
        # Check for NaN values in the input
        nmp.detect_nan(x, "DualPathBlock input")

        # Check noise map for NaN values if provided
        if noise_map is not None:
            nmp.detect_nan(noise_map, "DualPathBlock noise map")

        # shared feature extraction
        feat = self.features(x)
        feat = self.activation(feat)

        #----------------------------- DETAIL PATH -----------------------------#
        if self.use_texture_in_detail and texture_mask is not None:
            if self.use_noise_map and noise_map is not None:
                detail = nmp.apply_to_module(
                    feat, self.detail_path, noise_map, texture_mask
                )
            else:
                detail = nmp.apply_to_module(
                    feat, self.detail_path, texture_mask=texture_mask
                )
        else:
            if self.use_noise_map and noise_map is not None:
                detail = nmp.apply_to_module(
                    feat, self.detail_path, noise_map
                )
            else:
                detail = self.detail_path(feat)
        nmp.detect_nan(detail, "detail path output")

        #--------------------------- DENOISING PATH ---------------------------#
        # Compute adaptive gating mechanism
        if self.use_texture_in_denoise and texture_mask is not None:
            if self.use_noise_map and noise_map is not None:
                denoise = nmp.apply_to_module(
                    feat, self.denoise_path, noise_map, texture_mask
                )
            else:
                denoise = nmp.apply_to_module(
                    feat, self.denoise_path, texture_mask=texture_mask
                )
        else:
            if self.use_noise_map and noise_map is not None:
                # Get denoising gate with adjusted noise map
                denoise = nmp.apply_to_module(
                    feat, self.denoise_path, noise_map
                )
            else:
                # Get denoising gate without noise map
                denoise = self.denoise_path(feat)

        # Apply residual denoising with gating
        nmp.detect_nan(denoise, "denoising path output")

        #--------------------------- DYNAMIC FUSION ---------------------------#
        # Fuse outputs from both paths using content-aware mechanism
        if self.use_texture_in_fusion and texture_mask is not None:
            if self.use_noise_map and noise_map is not None:
                fusion_func = lambda d, n, t: self.fusion(d, denoise, feat, n, t)
                output = nmp.apply_to_module(
                    detail, fusion_func, noise_map, texture_mask, feat.size(1)
                )
            else:
                fusion_func = lambda d, t: self.fusion(d, denoise, feat, None, t)
                output = nmp.apply_to_module(
                    detail, fusion_func, texture_mask=texture_mask, target_channels=feat.size(1)
                )
        else:
            if self.use_noise_map and noise_map is not None:
                fusion_func = lambda d, n: self.fusion(d, denoise, feat, n, None)
                output = nmp.apply_to_module(
                    detail, fusion_func, noise_map=noise_map, target_channels=feat.size(1)
                )
            else:
                output = self.fusion(detail, denoise, feat)

        nmp.detect_nan(output, "fusion output")
        return output, detail, denoise  # 修改：返回融合输出和两条路径的独立输出

@ARCH_REGISTRY.register()
class DualPathUNet_E1_fusion(nn.Module):
    """double path U-Net, apply double path design on each scale of U-Net"""
    def __init__(self, in_channels=4, out_channels=4, base_channels=64, heads=[1,2,4,8],
                 dilated_rates=None, use_wavelet_upsample=True,
                 use_sharpness_recovery=True, use_noise_map=False,
                 use_texture_detection=False,  # 总开关，为了向后兼容
                 use_texture_in_detail=None,
                 use_texture_in_denoise=None,
                 use_texture_in_fusion=None,
                 use_texture_in_recovery=None,
                 texture_params=None,
                 enable_intermediate_supervision=True):  # 新增：中间监督开关
        super(DualPathUNet_E1_fusion, self).__init__()

        self.base_channels = base_channels
        self.use_noise_map = use_noise_map
        self.use_wavelet_upsample = use_wavelet_upsample
        self.use_sharpness_recovery = use_sharpness_recovery
        self.enable_intermediate_supervision = enable_intermediate_supervision  # 新增

        self.use_texture_detection = use_texture_detection
        self.use_texture_in_detail = use_texture_in_detail if use_texture_in_detail is not None else use_texture_detection
        self.use_texture_in_denoise = use_texture_in_denoise if use_texture_in_denoise is not None else use_texture_detection
        self.use_texture_in_fusion = use_texture_in_fusion if use_texture_in_fusion is not None else use_texture_detection
        self.use_texture_in_recovery = use_texture_in_recovery if use_texture_in_recovery is not None else use_texture_detection

        # enc1_in_channels = in_channels * 2 if use_noise_map else in_channels
        enc1_in_channels = in_channels

        # texture_detector
        self.texture_params = {
            'texture_gate': 0.5,
            'texture_suppress_factor': 0.7,
            'fusion_texture_boost': 0.5,
            'sharpness_texture_boost': 0.3,
        }
        if use_texture_detection:
            if texture_params is not None:
                self.texture_params.update(texture_params)

            # texture_detector params
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

        # 修改：替换编码器为独立路径编码器
        # 细节路径编码器
        self.enc1_detail = IndependentPathEncoder(
            enc1_in_channels, base_channels,
            is_detail_path=True,
            use_noise_map=use_noise_map,
            use_texture_detection=self.use_texture_in_detail,
            heads=heads[0],
            texture_params=self.texture_params
        )
        self.enc2_detail = IndependentPathEncoder(
            base_channels, base_channels*2,
            is_detail_path=True,
            use_noise_map=use_noise_map,
            use_texture_detection=self.use_texture_in_detail,
            heads=heads[1],
            texture_params=self.texture_params
        )
        self.enc3_detail = IndependentPathEncoder(
            base_channels*2, base_channels*4,
            is_detail_path=True,
            use_noise_map=use_noise_map,
            use_texture_detection=self.use_texture_in_detail,
            heads=heads[2],
            texture_params=self.texture_params
        )

        # 降噪路径编码器
        self.enc1_denoise = IndependentPathEncoder(
            enc1_in_channels, base_channels,
            is_detail_path=False,
            use_noise_map=use_noise_map,
            use_texture_detection=self.use_texture_in_denoise,
            heads=heads[0],
            texture_params=self.texture_params
        )
        self.enc2_denoise = IndependentPathEncoder(
            base_channels, base_channels*2,
            is_detail_path=False,
            use_noise_map=use_noise_map,
            use_texture_detection=self.use_texture_in_denoise,
            heads=heads[1],
            texture_params=self.texture_params
        )
        self.enc3_denoise = IndependentPathEncoder(
            base_channels*2, base_channels*4,
            is_detail_path=False,
            use_noise_map=use_noise_map,
            use_texture_detection=self.use_texture_in_denoise,
            heads=heads[2],
            texture_params=self.texture_params
        )

        # 瓶颈层 - 这里开始两条路径融合
        self.bottleneck = DualPathBlock(
            base_channels*4, base_channels*8,
            use_noise_map,
            self.use_texture_in_detail,
            self.use_texture_in_denoise,
            self.use_texture_in_fusion,
            self.texture_params,
            heads[3]
        )

        # decoder with skip connections - 保持现有结构
        self.dec3 = DualPathBlock(base_channels*4+base_channels*4, base_channels*4, use_noise_map, self.use_texture_in_detail, self.use_texture_in_denoise, self.use_texture_in_fusion, self.texture_params, heads[2])
        self.dec2 = DualPathBlock(base_channels*2+base_channels*2, base_channels*2, use_noise_map, self.use_texture_in_detail, self.use_texture_in_denoise, self.use_texture_in_fusion, self.texture_params, heads[1])
        self.dec1 = DualPathBlock(base_channels+base_channels, base_channels, use_noise_map, self.use_texture_in_detail, self.use_texture_in_denoise, self.use_texture_in_fusion, self.texture_params, heads[0])

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

        # 新增：中间监督输出层
        if enable_intermediate_supervision:
            # 细节路径最终输出层
            self.detail_output = nn.Conv2d(base_channels, out_channels, 1)
            # 降噪路径最终输出层
            self.denoise_output = nn.Conv2d(base_channels, out_channels, 1)

        # sharpness recovery
        if use_sharpness_recovery:
            self.sharpness_recovery = SharpnessRecovery(out_channels, use_noise_map, self.use_texture_in_recovery, sharpness_texture_boost=texture_params.get('sharpness_texture_boost', 0.3))

    def forward(self, x, noise_map=None, texture_mask=None):
        nmp.detect_nan(x, "input image")

        if self.use_texture_detection:
            computed_texture_mask = self.texture_detector(x, noise_map)
            if texture_mask is None:
                texture_mask = computed_texture_mask
            texture_mask = nmp.standardize_map(texture_mask)
            nmp.detect_nan(texture_mask, "纹理掩码")

            texture_masks = nmp.create_multiscale_maps(
                texture_mask, scales=[1, 2, 4, 6]
            )
            texture_masks = {
                'original': texture_masks['scale_1'],
                'down1': texture_masks['scale_2'],
                'down2': texture_masks['scale_4'],
                'down3': texture_masks['scale_6']
            }
        else:
            texture_mask = None
            texture_masks = {k: None for k in ['original', 'down1', 'down2', 'down3']}

        # noise maps
        if self.use_noise_map and noise_map is not None:
            noise_map = nmp.standardize_map(noise_map)
            noise_maps = nmp.create_multiscale_maps(
                noise_map, scales=[1, 2, 4, 6]
            )
            noise_maps = {
                'original': noise_maps['scale_1'],
                'down1': noise_maps['scale_2'],
                'down2': noise_maps['scale_4'],
                'down3': noise_maps['scale_6']
            }
            x_input = x
        else:
            noise_maps = {k: None for k in ['original', 'down1', 'down2', 'down3']}
            x_input = x

        #--------------------------- 修改：独立编码器路径 ---------------------------#

        # 细节路径编码
        enc1_detail = self.enc1_detail(x_input, noise_maps['original'], texture_masks['original'])
        enc1_detail_down = self.down(enc1_detail)
        enc2_detail = self.enc2_detail(enc1_detail_down, noise_maps['down1'], texture_masks['down1'])
        enc2_detail_down = self.down(enc2_detail)
        enc3_detail = self.enc3_detail(enc2_detail_down, noise_maps['down2'], texture_masks['down2'])
        enc3_detail_down = self.down(enc3_detail)

        # 降噪路径编码
        enc1_denoise = self.enc1_denoise(x_input, noise_maps['original'], texture_masks['original'])
        enc1_denoise_down = self.down(enc1_denoise)
        enc2_denoise = self.enc2_denoise(enc1_denoise_down, noise_maps['down1'], texture_masks['down1'])
        enc2_denoise_down = self.down(enc2_denoise)
        enc3_denoise = self.enc3_denoise(enc2_denoise_down, noise_maps['down2'], texture_masks['down2'])
        enc3_denoise_down = self.down(enc3_denoise)

        # 瓶颈层 - 合并两条路径
        # 将两个独立路径的特征连接起来输入到瓶颈层
        bottleneck_input = (enc3_detail_down + enc3_denoise_down) / 2  # 简单平均或者其他融合策略
        bottleneck_output, bn_detail, bn_denoise = self.bottleneck(
            bottleneck_input, noise_maps['down3'], texture_masks['down3']
        )

        #--------------------------- 解码器路径 ---------------------------#
        # 从这里开始使用双路径融合块

        # 解码器3
        bottleneck_up = self.up3(bottleneck_output)
        # 将跳跃连接从两条独立路径连接
        dec3_input = torch.cat([bottleneck_up, (enc3_detail + enc3_denoise)/2], dim=1)
        dec3_output, dec3_detail, dec3_denoise = self.dec3(
            dec3_input, noise_maps['down2'], texture_masks['down2']
        )

        # 解码器2
        dec3_up = self.up2(dec3_output)
        dec2_input = torch.cat([dec3_up, (enc2_detail + enc2_denoise)/2], dim=1)
        dec2_output, dec2_detail, dec2_denoise = self.dec2(
            dec2_input, noise_maps['down1'], texture_masks['down1']
        )

        # 解码器1
        dec2_up = self.up1(dec2_output)
        dec1_input = torch.cat([dec2_up, (enc1_detail + enc1_denoise)/2], dim=1)
        dec1_output, dec1_detail, dec1_denoise = self.dec1(
            dec1_input, noise_maps['original'], texture_masks['original']
        )

        #--------------------------- 新增：中间监督输出 ---------------------------#

        main_output = self.final(dec1_output)
        nmp.detect_nan(main_output, "final network output")

        # 可选的中间监督输出
        detail_output = None
        denoise_output = None

        if self.enable_intermediate_supervision:
            detail_output = self.detail_output(dec1_detail)
            denoise_output = self.denoise_output(dec1_denoise)

            # detail_output = torch.tanh(detail_output) * 0.5 + 0.5
            detail_output = torch.clamp(detail_output, 0.0, 1.0)
            denoise_output = torch.clamp(denoise_output, 0.0, 1.0)

            nmp.detect_nan(detail_output, "detail path output")
            nmp.detect_nan(denoise_output, "denoise path output")

        # sharpness recovery
        if self.use_sharpness_recovery:
            if self.use_texture_in_recovery and computed_texture_mask is not None:
                main_output = nmp.apply_to_module(
                    main_output, self.sharpness_recovery, noise_maps['original'], texture_masks['original']
                )
            else:
                main_output = nmp.apply_to_module(
                    main_output, self.sharpness_recovery, noise_maps['original']
                )

        # 返回主输出、纹理掩码和中间监督输出
        return main_output, texture_mask, detail_output, denoise_output