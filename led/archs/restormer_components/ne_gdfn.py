import torch
import torch.nn as nn
import torch.nn.functional as F

class NoiseEnhancedGDFN(nn.Module):
    def __init__(self, channels, expansion_factor=2.66):
        super(NoiseEnhancedGDFN, self).__init__()

        hidden_channels = int(channels * expansion_factor)

        # 基本GDFN组件
        self.project_in = nn.Conv2d(channels, hidden_channels*2, kernel_size=1)
        self.dwconv = nn.Conv2d(hidden_channels*2, hidden_channels*2, kernel_size=3,
                              padding=1, groups=hidden_channels*2)
        self.project_out = nn.Conv2d(hidden_channels, channels, kernel_size=1)

        # 噪声强度自适应分支
        self.noise_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(1, channels//4, kernel_size=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels//4, channels, kernel_size=1),
            nn.Sigmoid()
        )

        # 深层噪声滤波器
        self.noise_filter = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=5, padding=2, groups=channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1)
        )

    def forward(self, x, noise_map=None):
        # 保存原始输入
        identity = x

        # 投影到更高维度
        x1 = self.project_in(x)
        x1 = self.dwconv(x1)

        # 分离门控特征和内容特征
        x2, x3 = x1.chunk(2, dim=1)

        # 基本GDFN处理
        # 使用GELU激活函数的门控机制
        x4 = F.gelu(x2) * x3

        # 输出投影
        out = self.project_out(x4)

        # 噪声自适应处理
        if noise_map is not None:
            # 生成通道级别的噪声调制因子
            noise_strength = self.noise_gate(noise_map)

            # 强噪声区域使用强力滤波
            strong_filter = self.noise_filter(identity)

            # 自适应融合：噪声大的区域偏向强力滤波，噪声小的区域偏向常规处理
            # 注意这里提高了noise_strength的权重，使得噪声区域得到更强的处理
            out = out * (1.0 - noise_strength) + strong_filter * noise_strength

        return out