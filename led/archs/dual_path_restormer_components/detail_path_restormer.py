import torch
import torch.nn as nn
import torch.nn.functional as F
from ..restormer_components import MDTA

class EnhancedDetailPath(nn.Module):
    """增强的细节保留路径

    结合DilatedConvChain和MDTA
    增强边缘和纹理细节保留能力
    """

    def __init__(self, channels, dilated_rates=None, num_heads=1, use_noise_map=False):
        """初始化

        Args:
            channels: 输入特征通道数
            dilated_rates: 空洞卷积的膨胀率列表
            num_heads: MDTA中的头数
            use_noise_map: 是否使用噪声图
        """
        super(EnhancedDetailPath, self).__init__()

        if dilated_rates is None:
            dilated_rates = [1, 2, 4, 8]

        self.use_noise_map = use_noise_map

        # 空洞卷积链
        self.dilated_convs = nn.ModuleList([
            nn.Conv2d(channels, channels, 3, padding=r, dilation=r, bias=False)
            for r in dilated_rates
        ])

        # 激活函数
        self.activation = nn.LeakyReLU(0.2, inplace=True)

        # 多Dconv头转置注意力
        self.mdta = MDTA(channels, num_heads=num_heads)

        # Sobel边缘检测
        self.edge_detector = self._make_sobel_detector()

        # 边缘特征处理
        self.edge_conv = nn.Conv2d(channels, channels, 3, padding=1, bias=False)

        # 注意力图生成
        self.attention = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, padding=1, bias=False),
            nn.Sigmoid()
        )

        # 用于噪声图处理的卷积
        if use_noise_map:
            self.noise_conv = nn.Conv2d(1, channels, 1, bias=False)

    def _make_sobel_detector(self):
        """创建Sobel边缘检测器"""
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)

        sobel_x = sobel_x.view(1, 1, 3, 3).repeat(1, 1, 1, 1)
        sobel_y = sobel_y.view(1, 1, 3, 3).repeat(1, 1, 1, 1)

        return nn.ModuleList([
            nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False),
            nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        ])

    def forward(self, x, noise_map=None):
        """前向传播

        Args:
            x: 输入特征 [B, C, H, W]
            noise_map: 可选的噪声图 [B, 1, H, W]

        Returns:
            增强的特征 [B, C, H, W]
        """
        # 空洞卷积链
        res = x
        for conv in self.dilated_convs:
            res = self.activation(conv(res) + res)  # 残差连接

        # 应用MDTA获取全局上下文
        res = self.mdta(res)

        # 边缘检测
        B, C, H, W = x.shape
        x_reshaped = x.view(B*C, 1, H, W)

        edge_x = F.conv2d(x_reshaped, self.edge_detector[0].weight, padding=1)
        edge_y = F.conv2d(x_reshaped, self.edge_detector[1].weight, padding=1)

        edge = torch.sqrt(edge_x**2 + edge_y**2 + 1e-8)
        edge = edge.view(B, C, H, W)

        # 处理边缘特征
        edge_feat = self.edge_conv(edge)

        # 生成注意力图
        attention = self.attention(torch.cat([res, edge_feat], dim=1))

        # 如果有噪声图，调整注意力
        if self.use_noise_map and noise_map is not None:
            noise_feat = self.noise_conv(noise_map)
            # 高噪声区域减少边缘敏感度
            noise_weight = torch.exp(-5.0 * noise_map)
            attention = attention * noise_weight

        # 特征增强
        output = x * attention + res

        return output