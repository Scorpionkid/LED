import torch
import torch.nn as nn
import torch.nn.functional as F

class RAWTextureDetector(nn.Module):
    """RAW图像纹理检测器 - 适配RAW图像的4通道输入

    Args:
        window_sizes (list): 用于多尺度纹理检测的窗口大小列表
        base_lower_thresh (float): 标准差的基础低阈值
        base_upper_thresh (float): 标准差的基础高阈值
        adaptive_thresh (bool): 是否使用自适应阈值
        raw_channels (int): RAW图像的通道数，默认为4 (RGGB)
    """
    def __init__(self, window_sizes=[5, 9, 15], base_lower_thresh=0.05,
                 base_upper_thresh=0.2, adaptive_thresh=True, raw_channels=4):
        super(RAWTextureDetector, self).__init__()
        self.window_sizes = window_sizes
        self.base_lower_thresh = base_lower_thresh
        self.base_upper_thresh = base_upper_thresh
        self.adaptive_thresh = adaptive_thresh
        self.raw_channels = raw_channels

        # RAW图像特征提取，4通道输入
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(raw_channels, 32, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 8, 3, padding=1),
        )

        # 通道特定权重 - 学习不同颜色通道的重要性
        self.channel_weights = nn.Parameter(torch.ones(raw_channels) / raw_channels)

        # 多尺度融合
        self.scale_fusion = nn.Conv2d(len(window_sizes), 1, 1)

    def forward(self, x, noise_map=None):
        """前向传播

        Args:
            x (Tensor): 输入RAW图像 [B, 4, H, W]
            noise_map (Tensor, optional): 噪声图，如果有的话

        Returns:
            texture_mask (Tensor): 纹理掩码 [B, 1, H, W]，值范围[0,1]
        """
        # 提取特征
        features = self.feature_extractor(x)

        # 多尺度纹理检测
        texture_maps = []
        for window_size in self.window_sizes:
            # 计算当前尺度的标准差图
            std_map = self._compute_std_map(features, window_size)
            texture_maps.append(std_map)

        # 融合多尺度纹理图
        if len(self.window_sizes) > 1:
            multi_scale_texture = torch.cat(texture_maps, dim=1)
            fused_std_map = self.scale_fusion(multi_scale_texture)
        else:
            fused_std_map = texture_maps[0]

        # 获取阈值 - 固定或自适应
        if self.adaptive_thresh:
            lower_thresh, upper_thresh = self._compute_adaptive_thresholds(x, fused_std_map)
        else:
            lower_thresh, upper_thresh = self.base_lower_thresh, self.base_upper_thresh

        # 生成纹理掩码
        texture_mask = self._generate_texture_mask(fused_std_map, lower_thresh, upper_thresh)

        # 根据噪声图调整纹理掩码（如果有）
        if noise_map is not None:
            # 在高噪声区域降低纹理检测的灵敏度
            noise_factor = torch.exp(-3.0 * noise_map)  # 高噪声区域接近0，低噪声区域接近1
            texture_mask = texture_mask * noise_factor

        return texture_mask

    def _compute_std_map(self, features, window_size):
        """计算特征的局部标准差图"""
        N, C, H, W = features.shape
        pad = window_size // 2

        # 填充边界
        features_padded = F.pad(features, [pad] * 4, mode='reflect')

        # 使用unfold提取局部区域
        patches = F.unfold(features_padded, kernel_size=window_size)
        patches = patches.view(N, C, window_size * window_size, H, W)

        # 计算标准差
        mean = torch.mean(patches, dim=2, keepdim=True)
        var = torch.mean((patches - mean) ** 2, dim=2)
        std = torch.sqrt(var + 1e-8)  # 添加小值防止数值不稳定

        # 在通道维度上聚合
        std = torch.mean(std, dim=1, keepdim=True)

        return std

    def _compute_adaptive_thresholds(self, x, std_map):
        """计算自适应阈值"""
        # 基于全局标准差
        global_std = torch.std(x, dim=(2, 3), keepdim=True)
        # 将全局标准差映射到合理的阈值范围
        global_factor = torch.clamp(global_std * 5.0, 0.5, 2.0)

        # 调整基础阈值
        lower_thresh = self.base_lower_thresh * global_factor
        upper_thresh = self.base_upper_thresh * global_factor

        # 应用在通道维度上的权重
        channel_weights = F.softmax(self.channel_weights, dim=0).view(1, self.raw_channels, 1, 1)
        weighted_x = x * channel_weights
        channel_std = torch.std(weighted_x, dim=(2, 3), keepdim=True)

        # 进一步基于各通道的特性调整阈值
        channel_factor = torch.clamp(channel_std * 2.0, 0.8, 1.2)
        lower_thresh = lower_thresh * channel_factor
        upper_thresh = upper_thresh * channel_factor

        return lower_thresh, upper_thresh

    def _generate_texture_mask(self, std_map, lower_thresh, upper_thresh):
        """生成平滑过渡的纹理掩码"""
        # 标准化标准差
        normalized_std = (std_map - lower_thresh) / (upper_thresh - lower_thresh)
        normalized_std = torch.clamp(normalized_std, 0.0, 1.0)

        # 使用sigmoid函数实现平滑过渡
        texture_mask = torch.sigmoid(6.0 * normalized_std - 3.0)  # 调整斜率使过渡更加明显

        return texture_mask