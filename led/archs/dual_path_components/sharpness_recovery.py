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
    def __init__(self, in_channels=4, use_noise_map=False, use_texture_mask=False):
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
            self.texture_boost = nn.Parameter(torch.tensor(0.3))

    def forward(self, x, noise_map=None, texture_mask=None):

        if self.use_noise_map and noise_map is not None:
            # Use simple inversion and cropping functions.
            sharp_mask = torch.clamp(1.0 - noise_map * 5.0, 0.0, 1.0)
        else:
            noise_level = self.noise_estimator(x)
            sharp_mask = 1.0 - noise_level

        if self.use_texture_mask and texture_mask is not None:
            if texture_mask.shape[2:] != x.shape[2:]:
                raise ValueError(f"Texture mask shape {texture_mask.shape} does not match input shape {x.shape}")

            # 增强纹理区域的锐化强度
            texture_effect = self.texture_enhance(texture_mask)

            # 应用纹理增强（控制在合理范围内）
            boost_factor = torch.sigmoid(self.texture_boost)
            texture_boosted = sharp_mask + boost_factor * texture_effect * texture_mask
            sharp_mask = torch.clamp(texture_boosted, 0.0, 1.0)

        # Generate sharpness mask
        # sharpen in low noise regions, keep in high noise regions
        sharpened = self.adaptive_sharp(x)
        return sharpened * sharp_mask + x * (1.0 - sharp_mask)