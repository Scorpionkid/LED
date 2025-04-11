import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
class NoiseAdaptiveMDTA(nn.Module):
    """噪声自适应多Dconv头转置注意力

    根据噪声水平动态调整注意力计算方式
    """

    def __init__(self, channels, num_heads=1, bias=False):
        super(NoiseAdaptiveMDTA, self).__init__()

        self.num_heads = num_heads
        # 可学习的温度参数，现在是噪声自适应的
        self.base_temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        # 噪声调制模块
        self.noise_modulation = nn.Sequential(
            nn.Conv2d(1, channels//4, kernel_size=1, bias=bias),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels//4, num_heads, kernel_size=1, bias=bias),
            nn.Sigmoid()
        )

        # 使用统一的QKV投影
        self.qkv = nn.Conv2d(channels, channels*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(channels*3, channels*3, kernel_size=3,
                                  stride=1, padding=1, groups=channels*3, bias=bias)
        self.output_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=bias)

    def forward(self, x, noise_map=None):
        b, c, h, w = x.shape

        # 基础MDTA流程
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        # 重塑张量用于多头处理
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        # 应用L2归一化
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        # 噪声自适应温度调节
        if noise_map is not None:
            # 噪声调制因子，为每个头生成不同权重
            noise_weight = self.noise_modulation(noise_map)  # B,heads,H,W
            noise_weight = noise_weight.view(b, self.num_heads, 1, 1)

            # 动态温度：噪声大的区域使用较低温度，增强相关区域的筛选能力
            adaptive_temperature = self.base_temperature * (2.0 - noise_weight)
        else:
            adaptive_temperature = self.base_temperature

        # 计算注意力分数 - 在通道维度而非空间维度，使用动态温度
        attn = (q @ k.transpose(-2, -1)) * adaptive_temperature
        attn = attn.softmax(dim=-1)

        # 应用注意力权重
        out = (attn @ v)

        # 重塑回原始形状
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        # 输出投影
        out = self.output_proj(out)

        return out