import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveDenoiseGate(nn.Module):
    """Adaptive gating mechanism to adjust denoising strength based on local noise characteristics"""
    def __init__(self, channels, use_noise_map=False, use_texture_mask=False):
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
            # 纹理与噪声结合模块
            self.texture_noise_fusion = nn.Sequential(
                nn.Conv2d(2, 8, 3, padding=1),  # 输入：噪声估计+纹理掩码
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(8, 1, 3, padding=1),
                nn.Sigmoid()
            )
            # 纹理抑制因子（降低纹理区域的去噪强度）
            self.texture_suppress_factor = nn.Parameter(torch.tensor(0.7))  # 初始值0.7
    def forward(self, x, noise_map=None, texture_mask=None):
        """
        Args:
            x: input features
            noise_map: optional
        """
        # Estimate noise level
        if self.use_noise_map and noise_map is not None:
            denoise_strength = torch.sigmoid(5.0 * noise_map)

            if denoise_strength.size(1) == 1 and x.size(1) > 1:
                denoise_strength = denoise_strength.repeat(1, x.size(1), 1, 1)

            # 添加虚拟计算确保所有参数参与计算图
            dummy_noise = self.noise_est(x)
            dummy_gate = self.gate(dummy_noise)
            # 添加一个极小的影响，确保参数参与但不改变结果
            denoise_strength = denoise_strength + dummy_gate * 0.0
        else:
            estimated_noise = self.noise_est(x)
            denoise_strength = self.gate(estimated_noise)

        if self.use_texture_mask and texture_mask is not None:
            if texture_mask.shape[2:] != x.shape[2:]:
                raise ValueError(f"Texture mask shape {texture_mask.shape} does not match input shape {x.shape}")

            if self.use_noise_map and noise_map is not None:
                if noise_map.shape[2:] != x.shape[2:]:
                    raise ValueError(f"Noise map shape {noise_map.shape} does not match input shape {x.shape}")

                # 融合估计的噪声和纹理信息
                noise_texture = torch.cat([noise_map, texture_mask], dim=1)
                fusion_factor = self.texture_noise_fusion(noise_texture)

                # 获取纹理抑制系数（确保在合理范围内）
                suppress_factor = torch.sigmoid(self.texture_suppress_factor)

                # 在高纹理区域降低去噪强度
                texture_adjusted = denoise_strength * (1.0 - suppress_factor * texture_mask * fusion_factor)
                denoise_strength = torch.clamp(texture_adjusted, 0.0, 1.0)
            else:
                suppress_factor = torch.sigmoid(self.texture_suppress_factor)
                texture_adjusted = denoise_strength * (1.0 - suppress_factor * texture_mask)
                denoise_strength = torch.clamp(texture_adjusted, 0.0, 1.0)

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