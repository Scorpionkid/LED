import torch
import torch.nn as nn
import torch.nn.functional as F

class DenoisePath(nn.Module):

    def __init__(self, channels, use_noise_map=False, use_texture_mask=False, texture_suppress_factor=0.7):
        super(DenoisePath, self).__init__()
        self.use_noise_map = use_noise_map
        self.use_texture_mask = use_texture_mask

        # 噪声估计
        self.noise_est = nn.Conv2d(channels, 1, 3, padding=1)

        # 自适应门控
        self.gate = nn.Sequential(
            nn.Conv2d(1, channels//2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels//2, channels, 3, padding=1),
            nn.Sigmoid()
        )

        # 残差降噪器
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.activation = nn.ReLU(inplace=True)

        # 纹理抑制
        if use_texture_mask:
            self.texture_suppress_factor = nn.Parameter(torch.tensor(texture_suppress_factor))

    def forward(self, x, noise_map=None, texture_mask=None):
        """前向传播

        Args:
            x: 输入特征 [B, C, H, W]
            noise_map: 可选的噪声图 [B, 1, H, W]
            texture_mask: 可选的纹理掩码 [B, 1, H, W]

        Returns:
            降噪后的特征 [B, C, H, W]
        """
        # 估计噪声
        estimated_noise = self.noise_est(x)
        denoise_strength = self.gate(estimated_noise)
        base_strength = denoise_strength  # 保存基础值

        # 处理纹理掩码
        if self.use_texture_mask and texture_mask is not None:
            texture_weight = torch.clamp(torch.sigmoid(self.texture_suppress_factor), 0.2, 0.7)

            if self.use_noise_map and noise_map is not None:
                # 噪声调制
                noise_enhance = torch.sigmoid(4.0 * noise_map - 1.0)
                effective_weight = texture_weight * (1.0 - 0.6 * noise_enhance)
                denoise_strength = base_strength * (1.0 - effective_weight * texture_mask) + 0.2 * effective_weight * texture_mask
            else:
                # 无噪声图
                denoise_strength = base_strength * (1.0 - texture_weight * texture_mask) + 0.2 * texture_weight * texture_mask

        # 使用噪声图但无纹理掩码
        elif self.use_noise_map and noise_map is not None:
            noise_strength = 0.2 + 0.8 * torch.sigmoid(5.0 * noise_map)
            denoise_strength = 0.3 * base_strength + 0.7 * noise_strength

        # 限制降噪强度范围
        denoise_strength = torch.clamp(denoise_strength, 0.1, 0.9)

        # 计算残差
        residual = self.activation(self.conv1(x))
        residual = self.conv2(residual)

        # 应用降噪强度
        output = x + residual * denoise_strength

        return output