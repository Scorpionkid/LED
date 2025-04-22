import torch
import torch.nn as nn

class DynamicFusion(nn.Module):
    # Dynamic fusion layer, adaptively adjust fusion weights based on content complexity
    def __init__(self, channels, use_noise_map=False, use_texture_mask=False, fusion_texture_boost=0.5):
        super(DynamicFusion, self).__init__()
        self.use_noise_map = use_noise_map
        self.use_texture_mask = use_texture_mask

        # Complexity estimator
        # TODO:和cp1不一样
        self.complexity_est = nn.Sequential(
            nn.Conv2d(channels, channels//2, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels//2, channels//4, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels//4, 1, 3, padding=1),
            nn.Sigmoid()
        )

        if use_texture_mask:
            self.texture_boost_factor = nn.Parameter(torch.tensor(fusion_texture_boost))

        # 确保参数在合理范围
        self.register_buffer('min_factor', torch.tensor(0.0))
        self.register_buffer('max_factor', torch.tensor(1.0))

    def forward(self, detail_path, denoise_path, features, noise_map=None, texture_mask=None):
        """
        Args:
            detail_path: detail path output
            denoise_path: denoise path output
            features: features for complexity estimation
            texture_mask (Tensor, optional):
            noise_map (Tensor, optional):
        """
        # Estimate local complexity as fusion weight
        base_alpha = self.complexity_est(features)

        # TODO:结合噪声图进卷积or直接修正权重？
        if self.use_noise_map and noise_map is not None:

            # scheme1:
            # Regions with high noise should tend towards the denoising path,
            # while regions with rich details but low noise should tend towards the detail path.
            # input_features = torch.cat([features, noise_map], dim=1)
            # alpha = self.complexity_est(input_features)
            # # Reduce alpha (the weight of the detail path) in regions with high noise.
            # alpha = alpha * (1.0 - torch.sigmoid(noise_map * 3.0))

            # scheme2:
            noise_modifier = torch.exp(-5.0 * noise_map)
            alpha = base_alpha * noise_modifier

        else:
            alpha = base_alpha

        # Complex regions tend to detail path, flat regions tend to denoise path
        return alpha * detail_path + (1.0 - alpha) * denoise_path