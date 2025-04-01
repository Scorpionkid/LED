import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveDenoiseGate(nn.Module):
    """Adaptive gating mechanism to adjust denoising strength based on local noise characteristics"""
    def __init__(self, channels, use_noise_map=False, use_texture_mask=False, texture_suppress_factor=0.7,
                 texture_enhance_factor=0.3
    ):
        super(AdaptiveDenoiseGate, self).__init__()
        self.use_noise_map = use_noise_map
        self.use_texture_mask = use_texture_mask

        self.noise_est = nn.Conv2d(channels, 1, 3, padding=1)
        self.gate = nn.Sequential(
            nn.Conv2d(1, channels//2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels//2, channels, 3, padding=1),
            nn.Sigmoid()
        )

        if use_texture_mask:
            self.texture_suppress_factor = nn.Parameter(torch.tensor(texture_suppress_factor))
            self.texture_enhance_factor = nn.Parameter(torch.tensor(texture_enhance_factor))

    def forward(self, x, noise_map=None, texture_mask=None):
        """
        Args:
            x: input features
            noise_map: optional
        """
        # 基础降噪强度估计
        estimated_noise = self.noise_est(x)
        denoise_strength = self.gate(estimated_noise)
        base_strength = denoise_strength  # 保存基础值用于加权组合

        # 优先处理纹理相关逻辑
        if self.use_texture_mask and texture_mask is not None:
            suppress_factor = torch.clamp(torch.sigmoid(self.texture_suppress_factor), 0.3, 0.7)

            # 如果同时有噪声图，进行噪声感知的纹理处理
            if self.use_noise_map and noise_map is not None:
                # 噪声增强因子 - 高噪声区域接近1，低噪声区域接近0
                noise_enhance = torch.sigmoid(4.0 * noise_map - 1.0)

                # 加权组合：计算纹理区域的抑制量
                texture_suppress = suppress_factor * texture_mask

                # 基于噪声减少抑制效果（高噪声时需要更多降噪）
                noise_adjusted_suppress = texture_suppress * (1.0 - 0.6 * noise_enhance)

                # 应用抑制：基础值 - 调整后的抑制量
                denoise_strength = base_strength * (1.0 - noise_adjusted_suppress)
            else:
                # 原始思路但避免连乘
                texture_suppress = suppress_factor * texture_mask
                denoise_strength = base_strength * (1.0 - texture_suppress)

        # 噪声处理逻辑
        elif self.use_noise_map and noise_map is not None:
            # 使用基于噪声的直接强度，确保最小值
            noise_strength = 0.2 + 0.8 * torch.sigmoid(5.0 * noise_map)

            # 混合基础估计和噪声估计，保留基础特征
            denoise_strength = 0.3 * base_strength + 0.7 * noise_strength

        # 如果两者都没有，使用网络估计
        else:
            estimated_noise = self.noise_est(x)
            denoise_strength = self.gate(estimated_noise)

        denoise_strength = torch.clamp(denoise_strength, 0.1, 0.9)

        return denoise_strength

class ResidualDenoiser(nn.Module):
    """Residual learning structure to learn noise residuals"""
    def __init__(self, channels):
        super(ResidualDenoiser, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x, gate=None):
        """
        Args:
            x: input features
            gate: optional gating signal
        """
        residual = self.activation(self.conv1(x))
        residual = self.conv2(residual)

        if gate is not None:
            # If gating signal is provided, adjust residual based on gating signal
            return x + residual * gate
        else:
            return x + residual