import torch
import torch.nn as nn
import torch.nn.functional as F

class SobelFilter(nn.Module):
    """Sobel edge detection filter"""
    def __init__(self):
        super(SobelFilter, self).__init__()
        # Define Sobel filter
        self.sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                                    dtype=torch.float32).reshape(1, 1, 3, 3).repeat(1, 1, 1, 1)
        self.sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                                    dtype=torch.float32).reshape(1, 1, 3, 3).repeat(1, 1, 1, 1)

        # Register as buffer, so it will be saved in the model state, but not optimized as a parameter
        self.register_buffer('kernel_x', self.sobel_x)
        self.register_buffer('kernel_y', self.sobel_y)

    def forward(self, x):
        # Ensure the input dimension is correct
        b, c, h, w = x.shape
        x_reshaped = x.view(b*c, 1, h, w)

        # Apply Sobel filter
        edge_x = F.conv2d(x_reshaped, self.kernel_x, padding=1)
        edge_y = F.conv2d(x_reshaped, self.kernel_y, padding=1)

        # Calculate gradient magnitude
        edge = torch.sqrt(edge_x**2 + edge_y**2 + 1e-8)

        # Restore the original dimension
        return edge.view(b, c, h, w)

class DilatedConvChain(nn.Module):
    """Dilated convolution chain, using convolution cascades with different dilation rates to increase the receptive field"""
    def __init__(self, channels, dilated_rates=None):
        super(DilatedConvChain, self).__init__()
        if dilated_rates is None:
            dilated_rates = [1, 2, 4, 8]

        self.dilated_convs = nn.ModuleList([
            nn.Conv2d(channels, channels, 3, padding=r, dilation=r)
            for r in dilated_rates
        ])
        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        res = x
        for conv in self.dilated_convs:
            res = self.activation(conv(res) + res)  # Residual connection
        return res

class DetailPath(nn.Module):

    def __init__(self, channels, use_noise_map=False, use_texture_mask=False, texture_gate=0.5):
        super(DetailPath, self).__init__()
        self.use_noise_map = use_noise_map
        self.use_texture_mask = use_texture_mask

        # 膨胀卷积链
        self.dilated_convs = DilatedConvChain(channels)

        # 边缘检测
        self.edge_detector = SobelFilter()
        self.edge_conv = nn.Conv2d(channels, channels, 3, padding=1)

        # 注意力生成
        self.attention = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, padding=1),
            nn.Sigmoid()
        )

        # 纹理处理
        if use_texture_mask:
            self.texture_enhance = nn.Sequential(
                nn.Conv2d(channels + 1, channels, 1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(channels, channels, 3, padding=1)
            )
            self.texture_gate = nn.Parameter(torch.tensor(texture_gate))

    def forward(self, x, noise_map=None, texture_mask=None):
        """前向传播

        Args:
            x: 输入特征 [B, C, H, W]
            noise_map: 可选的噪声图 [B, 1, H, W]
            texture_mask: 可选的纹理掩码 [B, 1, H, W]

        Returns:
            增强的特征 [B, C, H, W]
        """
        # 膨胀卷积链处理
        feat = self.dilated_convs(x)

        # 边缘检测
        edge_map = self.edge_detector(x)
        edge_feat = self.edge_conv(edge_map)

        # 基本注意力
        base_attention = self.attention(torch.cat([feat, edge_feat], dim=1))

        # 纹理处理
        if self.use_texture_mask and texture_mask is not None:
            texture_feat = self.texture_enhance(torch.cat([feat, texture_mask], dim=1))
            gate = torch.clamp(self.texture_gate, 0.2, 0.8)

            if self.use_noise_map and noise_map is not None:
                # 保持原有噪声计算
                noise_factor = 0.4 + 0.6 * torch.exp(-4.0 * noise_map)

                effective_texture = texture_mask * noise_factor
                attention = base_attention * (1.0 - gate * effective_texture) + texture_feat * (gate * effective_texture)
            else:
                attention = base_attention

        # 噪声处理
        elif self.use_noise_map and noise_map is not None:
            noise_weight = torch.exp(-5.0 * noise_map)
            # noise_weight = 0.6 + 0.4 * torch.exp(-5.0 * noise_map)
            attention = base_attention * noise_weight
        else:
            attention = base_attention

        # 限制注意力范围
        attention = torch.clamp(attention, 0.1, 1.0)

        # 应用注意力
        return x * attention + x# Residual connection