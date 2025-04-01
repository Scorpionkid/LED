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

        # 优先处理纹理相关逻辑
        if self.use_texture_mask and texture_mask is not None:
            # 提取参数并限制范围
            texture_boost = torch.clamp(self.texture_boost_factor, 0.1, 0.8)
            smooth_suppress = torch.clamp(self.smooth_boost_factor, 0.1, 0.5)

            # 如果同时有噪声图
            if self.use_noise_map and noise_map is not None:
                # 噪声调制：高噪声区域减少细节保留
                noise_factor = torch.sigmoid(10.0 * (0.2 - noise_map))  # 阈值0.2可调整

                # 纹理区域增强（但受噪声调制）
                texture_alpha = base_alpha + texture_boost * texture_mask * noise_factor

                # 平滑区域抑制（但保留基本细节）
                smooth_regions = 1.0 - texture_mask
                smooth_alpha = base_alpha - smooth_suppress * smooth_regions

                # 加权组合：根据纹理掩码混合两种alpha
                alpha = texture_mask * texture_alpha + (1.0 - texture_mask) * smooth_alpha
            else:
                # 无噪声情况：直接纹理增强与平滑抑制
                texture_alpha = base_alpha + texture_boost * texture_mask
                smooth_regions = 1.0 - texture_mask
                smooth_alpha = base_alpha - smooth_suppress * smooth_regions

                alpha = texture_mask * texture_alpha + (1.0 - texture_mask) * smooth_alpha

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