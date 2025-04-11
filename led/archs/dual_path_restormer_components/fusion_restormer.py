import torch
import torch.nn as nn
import torch.nn.functional as F

class EnhancedFusion(nn.Module):
    """增强的动态融合层

    基于内容复杂度和噪声水平动态融合两条路径
    """

    def __init__(self, channels, use_noise_map=False, use_texture_mask=False, fusion_texture_boost=0.5):
        """初始化

        Args:
            channels: 输入特征通道数
            use_noise_map: 是否使用噪声图
            use_texture_mask: 是否使用纹理掩码
            fusion_texture_boost: 纹理增强因子
        """
        super(EnhancedFusion, self).__init__()

        self.use_noise_map = use_noise_map
        self.use_texture_mask = use_texture_mask

        # 内容复杂度估计
        self.complexity_est = nn.Sequential(
            nn.Conv2d(channels, channels//2, 3, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels//2, channels//4, 3, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels//4, 1, 3, padding=1, bias=False),
            nn.Sigmoid()
        )

        # 纹理增强
        if use_texture_mask:
            self.texture_boost_factor = nn.Parameter(torch.tensor(fusion_texture_boost))

        # 注册缓冲区
        self.register_buffer('min_factor', torch.tensor(0.0))
        self.register_buffer('max_factor', torch.tensor(1.0))

    def forward(self, detail_path, denoise_path, features, noise_map=None, texture_mask=None):
        """前向传播

        Args:
            detail_path: 细节路径输出 [B, C, H, W]
            denoise_path: 降噪路径输出 [B, C, H, W]
            features: 输入特征 [B, C, H, W]
            noise_map: 可选的噪声图 [B, 1, H, W]
            texture_mask: 可选的纹理掩码 [B, 1, H, W]

        Returns:
            融合后的特征 [B, C, H, W]
        """
        # 估计基础融合因子
        base_alpha = self.complexity_est(features)

        # 纹理掩码处理
        if self.use_texture_mask and texture_mask is not None:
            texture_weight = torch.clamp(self.texture_boost_factor, 0.1, 0.8)

            if self.use_noise_map and noise_map is not None:
                # 噪声因子
                noise_factor = torch.sigmoid(10.0 * (0.2 - noise_map))

                # 纹理影响
                texture_influence = texture_mask * noise_factor * texture_weight

                # 融合因子
                alpha = base_alpha * (1.0 - texture_influence) + texture_mask * texture_influence
            else:
                # 无噪声图
                texture_influence = texture_mask * texture_weight
                alpha = base_alpha * (1.0 - texture_influence) + texture_mask * texture_influence

        # 仅使用噪声图
        elif self.use_noise_map and noise_map is not None:
            # 噪声调制
            noise_modifier = 0.5 + 0.5 * torch.exp(-4.0 * noise_map)
            alpha = base_alpha * noise_modifier

        # 无噪声图和纹理掩码
        else:
            alpha = base_alpha

        # 限制融合因子范围
        alpha = torch.clamp(alpha, 0.1, 0.9)

        # 动态融合
        output = alpha * detail_path + (1.0 - alpha) * denoise_path

        return output