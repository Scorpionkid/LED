import torch
import torch.nn as nn
import torch.nn.functional as F
from led.utils.registry import ARCH_REGISTRY
import led.utils.noise_map_processor as nmp


from .dual_path_components import (
    DynamicFusion,
    WaveletUpsample, SharpnessRecovery, DiscreteWaveletUpsample,
    RAWTextureDetector,
    EnhancedDenoisePath, EnhancedDetailPath
)

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
        self.detail_path = EnhancedDetailPath(out_channels, num_heads, use_noise_map, use_texture_in_detail)

        self.denoise_path = EnhancedDenoisePath(out_channels, use_noise_map, use_texture_in_denoise, use_mdta=True)

        # dynamic fusion layer
        self.fusion = DynamicFusion(out_channels, use_noise_map, use_texture_in_fusion, fusion_texture_boost=fusion_texture_boost)

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
        return output

@ARCH_REGISTRY.register()
class DualPathUNet_E1(nn.Module):
    """double path U-Net, apply double path design on each scale of U-Net"""
    def __init__(self, in_channels=4, out_channels=4, base_channels=64, heads=[1,2,4,8],
                 dilated_rates=None, use_wavelet_upsample=True,
                 use_sharpness_recovery=True, use_noise_map=False,
                 use_texture_detection=False,  # 总开关，为了向后兼容
                 use_texture_in_detail=None,
                 use_texture_in_denoise=None,
                 use_texture_in_fusion=None,
                 use_texture_in_recovery=None,
                 texture_params=None):
        super(DualPathUNet_E1, self).__init__()

        self.base_channels = base_channels
        self.use_noise_map = use_noise_map
        self.use_wavelet_upsample = use_wavelet_upsample
        self.use_sharpness_recovery = use_sharpness_recovery

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
            # base_lower_thresh = texture_detector_params.get('base_lower_thresh', 0.05)
            # base_upper_thresh = texture_detector_params.get('base_upper_thresh', 0.2)
            adaptive_thresh = texture_detector_params.get('adaptive_thresh', True)
            noise_sensitivity = texture_detector_params.get('noise_sensitivity', 3.0)  # 新增

            self.texture_detector = RAWTextureDetector(
                window_sizes=window_sizes,
                # base_lower_thresh=base_lower_thresh,
                # base_upper_thresh=base_upper_thresh,
                adaptive_thresh=adaptive_thresh,
                raw_channels=in_channels,
                noise_sensitivity=noise_sensitivity
            )

        # encoder
        self.enc1 = DualPathBlock(enc1_in_channels, base_channels, use_noise_map, self.use_texture_in_detail, self.use_texture_in_denoise, self.use_texture_in_fusion, self.texture_params, heads[0])
        self.enc2 = DualPathBlock(base_channels, base_channels*2, use_noise_map, self.use_texture_in_detail, self.use_texture_in_denoise, self.use_texture_in_fusion, self.texture_params, heads[1])
        self.enc3 = DualPathBlock(base_channels*2, base_channels*4, use_noise_map, self.use_texture_in_detail, self.use_texture_in_denoise, self.use_texture_in_fusion, self.texture_params, heads[2])

        # bottleneck
        self.bottleneck = DualPathBlock(base_channels*4, base_channels*8, use_noise_map, self.use_texture_in_detail, self.use_texture_in_denoise, self.use_texture_in_fusion, self.texture_params, heads[3])

        # decoder with skip connections
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

        # sharpness recovery
        if use_sharpness_recovery:
            self.sharpness_recovery = SharpnessRecovery(out_channels, use_noise_map, self.use_texture_in_recovery, sharpness_texture_boost=texture_params.get('sharpness_texture_boost', 0.3))

    def forward(self, x, noise_map=None, texture_mask=None):
        # print("\n==== DUALPATHNET FORWARD ====")
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

        # concatenate noise map if needed
        if self.use_noise_map and noise_map is not None:
            # noise map for different size
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
            # x_input = torch.cat([x, noise_map], dim=1)
            x_input = x
        else:
            # print("Not using noise maps or noise_map is None")
            noise_maps = {k: None for k in ['original', 'down1', 'down2', 'down3']}
            x_input = x

        # print(f"x_input shape: {x_input.shape}")

        #--------------------------- ENCODER PATH ---------------------------#

        # [1, 4/8, 1024, 1024] -> [1, 64, 1024, 1024]
        enc1 = nmp.apply_to_module(
            x_input, self.enc1, noise_maps['original'], texture_masks['original']
        )
        # [1, 64, 1024, 1024] -> [1, 128, 512, 512]
        enc1_down = self.down(enc1)
        enc2 = nmp.apply_to_module(
            enc1_down, self.enc2, noise_maps['down1'], texture_masks['down1']
        )
        # [1, 128, 512, 512] -> [1, 256, 256, 256]
        enc2_down = self.down(enc2)
        enc3 = nmp.apply_to_module(
            enc2_down, self.enc3, noise_maps['down2'], texture_masks['down2']
        )

        #--------------------------- BOTTLENECK ---------------------------#

        # [1, 256, 256, 256] -> [1, 512, 128, 128]
        enc3_down = self.down(enc3)
        bottleneck = nmp.apply_to_module(
            enc3_down, self.bottleneck, noise_maps['down3'], texture_masks['down3']
        )
        nmp.detect_nan(bottleneck, "bottleneck")

        #--------------------------- DECODER PATH ---------------------------#

        # [1, 512, 128, 128] -> [[1, 256, 256, 256]] + [1, 256, 256, 256] -> [1, 256, 256, 256]
        bottleneck_up = self.up3(bottleneck)
        dec3_input = torch.cat([bottleneck_up, enc3], dim=1)
        dec3 = nmp.apply_to_module(
            dec3_input, self.dec3, noise_maps['down2'], texture_masks['down2']
        )
        # [1, 256, 256, 256] -> [1, 128, 512, 512]
        dec3_up = self.up2(dec3)
        dec2_input = torch.cat([dec3_up, enc2], dim=1)
        dec2 = nmp.apply_to_module(
            dec2_input, self.dec2, noise_maps['down1'], texture_masks['down1']
        )
        # [1, 128, 512, 512] -> [1, 64, 1024, 1024]
        dec2_up = self.up1(dec2)
        dec1_input = torch.cat([dec2_up, enc1], dim=1)
        dec1 = nmp.apply_to_module(
            dec1_input, self.dec1, noise_maps['original'], texture_masks['original']
        )

        #--------------------------- FINAL OUTPUT ---------------------------#

        out = self.final(dec1)
        nmp.detect_nan(out, "final network output")

        # sharpness recovery
        if self.use_sharpness_recovery:
            if self.use_texture_in_recovery and computed_texture_mask is not None:
                out = nmp.apply_to_module(
                    out, self.sharpness_recovery, noise_maps['original'], texture_masks['original']
                )
            else:
                out = nmp.apply_to_module(
                    out, self.sharpness_recovery, noise_maps['original']
                )

        return out, texture_mask