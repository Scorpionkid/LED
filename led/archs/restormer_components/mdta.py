import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class MDTA(nn.Module):
    """Multi-Dconv-Head Transposed Attention模块

    在特征维度而非空间维度上计算自注意力
    使用深度卷积进行局部上下文增强
    """

    def __init__(self, channels, num_heads=1, bias=False):
        """初始化

        Args:
            channels: 输入特征通道数
            num_heads: 注意力头数
            bias: 是否使用偏置
        """
        super(MDTA, self).__init__()

        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        # 使用统一的QKV投影
        self.qkv = nn.Conv2d(channels, channels*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(channels*3, channels*3, kernel_size=3,
                                  stride=1, padding=1, groups=channels*3, bias=bias)
        self.output_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=bias)

    def forward(self, x):
        """前向传播

        Args:
            x: 输入特征 [B, C, H, W]

        Returns:
            增强的特征 [B, C, H, W]
        """
        b, c, h, w = x.shape

        # 计算QKV投影
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        # 重塑张量用于多头处理
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        # 应用L2归一化
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        # 计算注意力分数 - 在通道维度而非空间维度
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        # 应用注意力权重
        out = (attn @ v)

        # 重塑回原始形状
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        # 输出投影
        out = self.output_proj(out)

        return out