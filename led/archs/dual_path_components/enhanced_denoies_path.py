import torch
import torch.nn as nn
import torch.nn.functional as F
from ..restormer_components.ne_gdfn import NoiseEnhancedGDFN as NE_GDFN

class AdaptiveDenoiseGate(nn.Module):
    """Adaptive gating mechanism to adjust denoising strength based on local noise characteristics"""
    def __init__(self, channels, use_noise_map=False, use_texture_mask=False, texture_suppress_factor=0.7):
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
    def forward(self, x, noise_map=None, texture_mask=None):
        """
        Args:
            x: input features
            noise_map: optional
        """
        # Estimate noise level
        if self.use_noise_map and noise_map is not None:
            denoise_strength = torch.sigmoid(5.0 * noise_map)

            # 添加虚拟计算确保所有参数参与计算图
            dummy_noise = self.noise_est(x)
            dummy_gate = self.gate(dummy_noise)
            # 添加一个极小的影响，确保参数参与但不改变结果
            denoise_strength = denoise_strength + dummy_gate * 0.0
        else:
            estimated_noise = self.noise_est(x)
            denoise_strength = self.gate(estimated_noise)

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

class EnhancedDenoisePath(nn.Module):
    def __init__(self, channels, use_noise_map=False):
        super(EnhancedDenoisePath, self).__init__()

        # 保留原有的自适应门控
        self.gate = AdaptiveDenoiseGate(channels, use_noise_map)

        # 新增：GDFN用于增强特征变换
        self.gdfn = NE_GDFN(channels)

        # 保留原有的残差降噪器
        self.residual_denoiser = ResidualDenoiser(channels)

    def forward(self, x, noise_map=None, texture_mask=None):
        # 生成自适应门控信号
        denoise_strength = self.gate(x, noise_map, texture_mask)

        # GDFN特征转换 - 增强降噪能力
        gdfn_features = self.gdfn(x, noise_map)

        # 结合GDFN与残差降噪
        # 先应用残差降噪
        residual_features = self.residual_denoiser(x)

        # 根据噪声程度调整GDFN和残差特征的比例
        if noise_map is not None:
            # 高噪声区域更依赖GDFN
            noise_weight = torch.sigmoid(3.0 * noise_map)
            output = x + denoise_strength * (
                gdfn_features * noise_weight + residual_features * (1 - noise_weight)
            )
        else:
            # 无噪声信息时平衡使用
            output = x + denoise_strength * (gdfn_features * 0.5 + residual_features * 0.5)

        return output