import torch
import torch.nn as nn

class DynamicFusion(nn.Module):
    # Dynamic fusion layer, adaptively adjust fusion weights based on content complexity
    def __init__(self, channels, use_noise_map=False, use_texture_mask=False, fusion_texture_boost=0.5
                 , fusion_smooth_boost=0.3
                 ):
        super(DynamicFusion, self).__init__()
        self.use_noise_map = use_noise_map
        self.use_texture_mask = use_texture_mask

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
            self.smooth_boost_factor = nn.Parameter(torch.tensor(fusion_smooth_boost))

        # 确保参数在合理范围
        self.register_buffer('min_factor', torch.tensor(0.0))
        self.register_buffer('max_factor', torch.tensor(1.0))

    def forward(self, detail_path, denoise_path, features, noise_map=None, texture_mask=None):

        base_alpha = self.complexity_est(features)

        if self.use_texture_mask and texture_mask is not None:
            # 单一参数控制纹理保留程度
            texture_weight = torch.clamp(self.texture_boost_factor, 0.1, 0.8)

            if self.use_noise_map and noise_map is not None:
                noise_factor = torch.sigmoid(10.0 * (0.2 - noise_map))

                # 简化的线性插值
                # - 高纹理+低噪声：保留更多细节
                # - 低纹理或高噪声：增强降噪
                texture_influence = texture_mask * noise_factor * texture_weight

                # 单一线性插值
                alpha = base_alpha * (1.0 - texture_influence) + texture_mask * texture_influence
            else:
                texture_influence = texture_mask * texture_weight
                alpha = base_alpha * (1.0 - texture_influence) + texture_mask * texture_influence

        # 噪声处理逻辑
        elif self.use_noise_map and noise_map is not None:
            # 噪声调制，但保留基本融合能力
            noise_modifier = 0.5 + 0.5 * torch.exp(-4.0 * noise_map)
            alpha = base_alpha * noise_modifier

        # 基础融合逻辑保持不变
        else:
            alpha = base_alpha

        # 确保融合因子在合理范围
        alpha = torch.clamp(alpha, 0.1, 0.9)


        return alpha * detail_path + (1.0 - alpha) * denoise_path