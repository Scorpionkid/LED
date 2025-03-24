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

class HighFrequencyAttention(nn.Module):
    # High-frequency attention module, focusing on edge and texture information in the image
    def __init__(self, channels, use_noise_map=False):
        super(HighFrequencyAttention, self).__init__()
        self.edge_detector = SobelFilter()
        self.conv_edge = nn.Conv2d(channels, channels, 3, padding=1)
        self.use_noise_map = use_noise_map

        self.attention = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x, noise_map = None):
        # Edge detection
        edge_map = self.edge_detector(x)
        edge_feat = self.conv_edge(edge_map)

        # base attention
        attention = self.attention(torch.cat([x, edge_feat], dim=1))

        # # Directly use the noise map to correct the attention.
        # Regions with high noise should reduce edge sensitivity.
        if self.use_noise_map and noise_map is not None:
            # The larger the value of the noise map, the smaller the attention weight.
            noise_weight = torch.exp(-5.0 * noise_map)  # Exponential decay function
            attention = attention * noise_weight

        return x * attention + x # Residual connection