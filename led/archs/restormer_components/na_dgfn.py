import torch
import torch.nn as nn
import torch.nn.functional as F

class NoiseAwareGDFN(nn.Module):
    """噪声感知门控Dconv前馈网络

    根据噪声水平动态调整特征转换强度和通道权重
    """

    def __init__(self, channels, expansion_factor=2.66, bias=False):
        super(NoiseAwareGDFN, self).__init__()

        hidden_channels = int(channels * expansion_factor)

        # 特征投影与转换
        self.project_in = nn.Conv2d(channels, hidden_channels*2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_channels*2, hidden_channels*2, kernel_size=3,
                              stride=1, padding=1, groups=hidden_channels*2, bias=bias)
        self.project_out = nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=bias)

        # 噪声调制模块 - 用于动态调整通道权重
        self.noise_channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(1, hidden_channels//2, kernel_size=1, bias=bias),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_channels//2, hidden_channels, kernel_size=1, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x, noise_map=None):
        # 投影到更高维度
        x1 = self.project_in(x)
        x1 = self.dwconv(x1)

        # 分离门控特征和内容特征
        x2, x3 = x1.chunk(2, dim=1)

        # 应用门控机制：内容特征 * (GELU激活的门控特征)
        x4 = F.gelu(x2) * x3

        # 噪声自适应通道注意力
        if noise_map is not None:
            # 生成通道级别的噪声调制因子
            channel_weights = self.noise_channel_attention(noise_map)

            # 根据噪声分布动态调整特征通道权重
            # 高噪声区域的特征会更多地偏向平滑处理
            x4 = x4 * channel_weights

        # 输出投影
        out = self.project_out(x4)

        return out