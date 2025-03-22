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
    def __init__(self, in_channels=4, use_noise_map=False):
        super(SharpnessRecovery, self).__init__()
        self.use_noise_map = use_noise_map
        if not use_noise_map:
            self.noise_estimator = NoiseLevelNetwork(in_channels)
        self.adaptive_sharp = AdaptiveUnsharpMask(in_channels)

    def forward(self, x, noise_map=None):

        if self.use_noise_map and noise_map is not None:
            noise_level = noise_map.mean(dim=1, keepdim=True)
        else:
            noise_level = self.noise_estimator(x)

        # Generate sharpness mask
        # sharpen in low noise regions, keep in high noise regions
        sharp_mask = 1 - noise_level
        sharpened = self.adaptive_sharp(x)
        return sharpened * sharp_mask + x * (1 - sharp_mask)