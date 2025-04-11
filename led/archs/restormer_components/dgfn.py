import torch
import torch.nn as nn
import torch.nn.functional as F

class GDFN(nn.Module):
    """Gated-Dconv Feed-Forward Network

    使用门控机制控制信息流
    结合深度卷积增强局部上下文
    """

    def __init__(self, channels, expansion_factor=2.66, bias=False):
        """初始化

        Args:
            channels: 输入特征通道数
            expansion_factor: 特征扩展因子
            bias: 是否使用偏置
        """
        super(GDFN, self).__init__()

        hidden_channels = int(channels * expansion_factor)

        # 特征投影与转换
        self.project_in = nn.Conv2d(channels, hidden_channels*2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_channels*2, hidden_channels*2, kernel_size=3,
                              stride=1, padding=1, groups=hidden_channels*2, bias=bias)
        self.project_out = nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=bias)

    def forward(self, x):
        """前向传播

        Args:
            x: 输入特征 [B, C, H, W]

        Returns:
            增强的特征 [B, C, H, W]
        """
        # 投影到更高维度
        x1 = self.project_in(x)
        x1 = self.dwconv(x1)

        # 分离门控特征和内容特征
        x2, x3 = x1.chunk(2, dim=1)

        # 应用门控机制：内容特征 * (GELU激活的门控特征)
        x4 = F.gelu(x2) * x3

        # 输出投影
        out = self.project_out(x4)

        return out