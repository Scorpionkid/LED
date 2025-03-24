import torch
import torch.nn as nn

class DynamicFusion(nn.Module):
    # Dynamic fusion layer, adaptively adjust fusion weights based on content complexity
    def __init__(self, channels, use_noise_map=False):
        super(DynamicFusion, self).__init__()
        self.use_noise_map = use_noise_map

        # Complexity estimator
        self.complexity_est = nn.Sequential(
            nn.Conv2d(input_channels, channels//2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels//2, 1, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, detail_path, denoise_path, features, noise_map=None):
        """
        Args:
            detail_path: detail path output
            denoise_path: denoise path output
            features: features for complexity estimation
        """
        # Estimate local complexity as fusion weight
        alpha = self.complexity_est(features)

        # TODO:结合噪声图进卷积or直接修正权重？
        if self.use_noise_map and noise_map is not None:

            # scheme1:
            # # Regions with high noise should tend towards the denoising path,
            # # while regions with rich details but low noise should tend towards the detail path.
            # input_features = torch.cat([features, noise_map], dim=1)
            # alpha = self.complexity_est(input_features)
            # # Reduce alpha (the weight of the detail path) in regions with high noise.
            # alpha = alpha * (1.0 - torch.sigmoid(noise_map * 3.0))

            # scheme2:
            noise_modifier = torch.exp(-5.0 * noise_map)
            alpha = alpha * noise_modifier

        # else:
        #     alpha = self.complexity_est(features)

        # Complex regions tend to detail path, flat regions tend to denoise path
        return alpha * detail_path + (1 - alpha) * denoise_path