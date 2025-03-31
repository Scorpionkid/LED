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
        # Estimate noise level
        if self.use_texture_mask and texture_mask is not None:
            # 如果有纹理掩码，我们应该直接用网络估计基础去噪强度
            # 因为纹理掩码已经包含了噪声信息
            estimated_noise = self.noise_est(x)
            denoise_strength = self.gate(estimated_noise)

            suppress_factor = torch.sigmoid(self.texture_suppress_factor)
            # enhance_factor = torch.sigmoid(self.texture_enhance_factor)

            texture_adjusted = denoise_strength * (1.0 - suppress_factor * texture_mask)
            # smooth_adjusted = denoise_strength * (1.0 + enhance_factor * (1.0 - texture_mask))

            denoise_strength = texture_mask * texture_adjusted + (1.0 - texture_mask)
            denoise_strength = torch.clamp(denoise_strength, 0.0, 1.0)

        # 只有在没有纹理掩码时才直接使用噪声图
        elif self.use_noise_map and noise_map is not None:
            denoise_strength = torch.sigmoid(5.0 * noise_map)

            # 添加虚拟计算确保所有参数参与计算图
            dummy_noise = self.noise_est(x)
            dummy_gate = self.gate(dummy_noise)
            # 添加一个极小的影响，确保参数参与但不改变结果
            denoise_strength = denoise_strength + dummy_gate * 0.0

        # 如果两者都没有，使用网络估计
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