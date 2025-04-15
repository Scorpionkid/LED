import torch
import torch.nn as nn
import torch.nn.functional as F
from ..restormer_components.ne_gdfn import NoiseEnhancedGDFN as NE_GDFN
from ..restormer_components.te_mdta import TextureEnhancedMDTA as TE_MDTA

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

class EnhancedResidualDenoiser(nn.Module):
    """改进的残差降噪器，融入GDFN的门控思想"""
    def __init__(self, channels):
        super(EnhancedResidualDenoiser, self).__init__()

        # 保留原有结构
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)

        # 引入门控机制（GDFN思想）- 替代原来的第二个卷积
        self.content_conv = nn.Conv2d(channels, channels, 3, padding=1)
        self.gate_conv = nn.Conv2d(channels, channels, 3, padding=1)

        self.activation = nn.ReLU(inplace=True)

    def forward(self, x, gate=None):
        """
        Args:
            x: 输入特征
            gate: 可选的外部门控信号
        """
        # 第一阶段与原来相同
        features = self.activation(self.conv1(x))

        # 分离门控特征和内容特征
        content = self.content_conv(features)
        internal_gate = torch.sigmoid(self.gate_conv(features))

        # 内外门控结合
        if gate is not None:
            effective_gate = gate * internal_gate
        else:
            effective_gate = internal_gate

        # 应用残差连接
        return x + content * effective_gate

class EnhancedDenoisePath(nn.Module):
    def __init__(self, channels, use_noise_map=False, use_texture_mask=False, use_mdta=True):
        super(EnhancedDenoisePath, self).__init__()

        # 保留原有的自适应门控
        self.gate = AdaptiveDenoiseGate(channels, use_noise_map)

        # 可选的MDTA全局上下文处理
        self.use_mdta = use_mdta
        if use_mdta:
            # 简化版MDTA，仅用于去噪路径的上下文增强
            self.pre_conv = nn.Conv2d(channels, channels, 1)
            self.mdta = TE_MDTA(channels, num_heads=2)
            self.post_conv = nn.Conv2d(channels, channels, 1)

        # 使用改进的残差降噪器
        self.residual_denoiser = EnhancedResidualDenoiser(channels)

    def forward(self, x, noise_map=None, texture_mask=None):
        # 生成自适应门控信号
        denoise_strength = self.gate(x, noise_map)

        # 可选的全局上下文增强
        if self.use_mdta:
            mdta_in = self.pre_conv(x)
            mdta_out = self.mdta(mdta_in, texture_mask)
            context_enhanced = x + 0.2 * self.post_conv(mdta_out)
        else:
            context_enhanced = x

        # 应用改进的残差降噪
        output = self.residual_denoiser(context_enhanced, denoise_strength)

        return output