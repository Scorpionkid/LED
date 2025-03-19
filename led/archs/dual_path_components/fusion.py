import torch
import torch.nn as nn

class DynamicFusion(nn.Module):
    # Dynamic fusion layer, adaptively adjust fusion weights based on content complexity
    def __init__(self, channels):
        super(DynamicFusion, self).__init__()
        # Complexity estimator
        self.complexity_est = nn.Sequential(
            nn.Conv2d(channels, channels//2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels//2, 1, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, detail_path, denoise_path, features):
        """
        Args:
            detail_path: detail path output
            denoise_path: denoise path output
            features: features for complexity estimation
        """
        # Estimate local complexity as fusion weight
        alpha = self.complexity_est(features)

        # Complex regions tend to detail path, flat regions tend to denoise path
        return alpha * detail_path + (1 - alpha) * denoise_path