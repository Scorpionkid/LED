import torch
import torch.nn as nn
import torch.nn.functional as F

class NoiseLevelNetwork(nn.Module):
    """Noise level estimation network"""
    def __init__(self, in_channels=4):
        super(NoiseLevelNetwork, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)

class AdaptiveUnsharpMask(nn.Module):
    """Adaptive sharpening module"""
    def __init__(self, in_channels=4):
        super(AdaptiveUnsharpMask, self).__init__()

        # Sharpening strength prediction
        self.sharpness_strength = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Predict sharpening strength
        strength = self.sharpness_strength(x)

        # Generate blurred version
        blurred = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)

        # Calculate high-frequency component
        high_freq = x - blurred

        # Add sharpening
        sharpened = x + high_freq * strength

        return sharpened


class SharpnessRecovery(nn.Module):
    def __init__(self, in_channels=4, use_noise_map=False, use_texture_mask=False, sharpness_texture_boost=0.3):
        super(SharpnessRecovery, self).__init__()
        self.use_noise_map = use_noise_map
        self.use_texture_mask = use_texture_mask
        if not use_noise_map:
            self.noise_estimator = NoiseLevelNetwork(in_channels)
        self.adaptive_sharp = AdaptiveUnsharpMask(in_channels)

        if use_texture_mask:
            self.texture_enhance = nn.Sequential(
                nn.Conv2d(1, 8, 3, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(8, 1, 3, padding=1),
                nn.Sigmoid()
            )
            self.texture_boost = nn.Parameter(torch.tensor(sharpness_texture_boost))

    def forward(self, x, noise_map=None, texture_mask=None):

        if self.use_texture_mask and texture_mask is not None:
            boost_factor = torch.sigmoid(self.texture_boost)

            # 使用网络增强纹理掩码的表现力
            enhanced_texture = self.texture_enhance(texture_mask)

            # - 高纹理区域(texture_mask接近1): 强锐化
            # - 低纹理区域(texture_mask接近0): 弱锐化
            sharp_mask = texture_mask * boost_factor * enhanced_texture

            sharp_mask = torch.clamp(sharp_mask, 0.0, 1.0)

        # 如果没有纹理掩码，使用噪声图或内部估计器
        elif self.use_noise_map and noise_map is not None:
            sharp_mask = torch.clamp(1.0 - noise_map * 5.0, 0.0, 1.0)
        else:
            noise_level = self.noise_estimator(x)
            sharp_mask = 1.0 - noise_level

        # Generate sharpness mask
        # sharpen in low noise regions, keep in high noise regions
        sharpened = self.adaptive_sharp(x)
        return sharpened * sharp_mask + x * (1.0 - sharp_mask)