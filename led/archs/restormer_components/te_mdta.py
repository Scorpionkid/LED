from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F


class TextureEnhancedMDTA(nn.Module):
    def __init__(self, channels, num_heads=1):
        super(TextureEnhancedMDTA, self).__init__()

        self.num_heads = num_heads
        # 分别为高频和低频内容使用不同的温度参数
        self.detail_temperature = nn.Parameter(torch.ones(num_heads, 1, 1) * 1.5)  # 更高的温度保留更多细节
        self.smooth_temperature = nn.Parameter(torch.ones(num_heads, 1, 1) * 0.5)  # 更低的温度平滑区域

        # 标准MDTA组件
        self.qkv = nn.Conv2d(channels, channels*3, kernel_size=1)
        self.qkv_dwconv = nn.Conv2d(channels*3, channels*3, kernel_size=3, padding=1, groups=channels*3)
        self.output_proj = nn.Conv2d(channels, channels, kernel_size=1)

        # 边缘检测器 - 用于识别纹理区域
        self.edge_detector = self._make_edge_detector()

    def _make_edge_detector(self):
        # 高通滤波器 - 突出边缘和纹理
        kernel = torch.tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=torch.float32).view(1, 1, 3, 3)
        return nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)

    def forward(self, x, noise_map=None):
        b, c, h, w = x.shape

        # 提取边缘图用于纹理区域识别
        edge_map = self._extract_edges(x)

        # 计算QKV投影
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        # 重塑张量
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        # 应用L2归一化
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        # 使用边缘图作为纹理权重，对每个位置使用自适应温度
        texture_weights = F.adaptive_avg_pool2d(edge_map, (1, 1))
        texture_weights = texture_weights.view(b, 1, 1, 1)

        # 动态温度: 高纹理区域使用高温度(保留细节)，低纹理区域使用低温度(平滑处理)
        adaptive_temperature = self.detail_temperature * texture_weights + self.smooth_temperature * (1 - texture_weights)

        # 计算注意力，使用自适应温度
        attn = (q @ k.transpose(-2, -1)) * adaptive_temperature
        attn = attn.softmax(dim=-1)

        # 如果有噪声图，进一步调整注意力
        if noise_map is not None:
            # 降低高噪声区域的纹理敏感度
            noise_weight = torch.exp(-2.0 * F.adaptive_avg_pool2d(noise_map, (1, 1)))
            noise_weight = noise_weight.view(b, 1, 1, 1)
            attn = attn * noise_weight + attn * (1 - noise_weight) * 0.8

        # 应用注意力
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        # 输出投影
        out = self.output_proj(out)

        # 加强边缘 - 按比例增强高频信息
        edge_enhance = edge_map * out * 0.2
        out = out + edge_enhance

        return out

    def _extract_edges(self, x):
        # 提取灰度图
        if x.size(1) > 1:
            gray = torch.mean(x, dim=1, keepdim=True)
        else:
            gray = x

        # 应用边缘检测
        edges = torch.abs(self.edge_detector(gray))

        # 归一化
        edges = edges / (torch.max(edges) + 1e-5)

        return edges