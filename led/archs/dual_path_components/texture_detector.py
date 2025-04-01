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
    def __init__(self, window_sizes=[11, 25, 49], base_lower_thresh=0.05,
                 base_upper_thresh=0.2, adaptive_thresh=True, raw_channels=4,
                 noise_sensitivity=3.0):
        super(RAWTextureDetector, self).__init__()
        self.window_sizes = window_sizes
        self.base_lower_thresh = base_lower_thresh
        self.base_upper_thresh = base_upper_thresh
        self.adaptive_thresh = adaptive_thresh
        self.raw_channels = raw_channels
        self.noise_sensitivity = nn.Parameter(torch.tensor(noise_sensitivity))

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
        if len(window_sizes) > 1:
            self.scale_fusion = nn.Sequential(
                nn.Conv2d(len(window_sizes), 1, 1),
                nn.ReLU()  # 确保输出非负
            )
            # 初始化为均匀权重
            nn.init.constant_(self.scale_fusion[0].weight, 1.0 / len(window_sizes))
            nn.init.zeros_(self.scale_fusion[0].bias)

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

        # # 根据噪声图调整纹理掩码（如果有）
        # if noise_map is not None:
        #     if noise_map.size(1) > 1:
        #         noise_map = torch.mean(noise_map, dim=1, keepdim=True)
        #     noise_sensitivity = torch.clamp(self.noise_sensitivity, 0.1, 5.0)
        #     noise_factor = torch.exp(-noise_sensitivity * noise_map)

        #     noise_factor = torch.clamp(noise_factor, 1e-6, 1.0)
        #     texture_mask = texture_mask * noise_factor

        return texture_mask

    # def _compute_std_map(self, features, window_size):
    #     """计算特征的局部标准差图"""
    #     N, C, H, W = features.shape
    #     pad = window_size // 2

    #     # 填充边界
    #     features_padded = F.pad(features, [pad] * 4, mode='reflect')

    #     # 使用unfold提取局部区域
    #     patches = F.unfold(features_padded, kernel_size=window_size)
    #     patches = patches.view(N, C, window_size * window_size, H, W)

    #     # 计算标准差
    #     mean = torch.mean(patches, dim=2, keepdim=True)
    #     var = torch.mean((patches - mean) ** 2, dim=2)
    #     std = torch.sqrt(var + 1e-8)  # 添加小值防止数值不稳定

    #     # 在通道维度上聚合
    #     std = torch.mean(std, dim=1, keepdim=True)

    #     return std

    def _compute_std_map(self, features, window_size):
        N, C, H, W = features.shape
        pad = window_size // 2

        features_padded = F.pad(features, [pad]*4, mode='reflect')

        avg_kernel = torch.ones(1, 1, window_size, window_size,
                            device=features.device) / (window_size**2)

        result = torch.zeros(N, 1, H, W, device=features.device, dtype=torch.float32)

        if C == 4:  # 原始RGGB输入
            channel_weights = torch.tensor([1.0, 1.5, 1.0, 1.5], device=features.device)
        else:  # 特征提取后的输入
            channel_weights = torch.ones(C, device=features.device)

        channel_weights = F.softmax(channel_weights, dim=0)

        for c in range(C):
            curr_feat = features_padded[:, c:c+1]

            local_mean = F.conv2d(curr_feat, avg_kernel, stride=1, padding=0, groups=1)

            local_mean_sq = F.conv2d(curr_feat**2, avg_kernel, stride=1, padding=0, groups=1)
            local_var = torch.clamp(local_mean_sq - local_mean**2, min=1e-6)

            result += channel_weights[c] * torch.sqrt(local_var)

        result = torch.pow(result, 0.8)

        return result

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
        """Generate texture mask using quantiles and segmented sigmoid

        Args:
            std_map: Standard deviation map [1, 1, H, W]
            lower_thresh: Lower threshold [1, 4, 1, 1]
            upper_thresh: Upper threshold [1, 4, 1, 1]

        Returns:
            Texture mask [1, 1, H, W]
        """
        # Compute quantiles for the standard deviation map
        std_flat = std_map.view(-1)
        q_low = torch.quantile(std_flat, 0.3)  # 30% quantile
        q_high = torch.quantile(std_flat, 0.7)  # 70% quantile

        # Since lower_thresh and upper_thresh are [1, 4, 1, 1] but std_map is [1, 1, H, W],
        # we need to convert the thresholds to match std_map's shape
        # Take average across channel dimension (dim=1)
        lower_mean = torch.mean(lower_thresh, dim=1, keepdim=True)  # [1, 1, 1, 1]
        upper_mean = torch.mean(upper_thresh, dim=1, keepdim=True)  # [1, 1, 1, 1]

        # Combine original thresholds with quantiles
        l_thresh = (lower_mean + q_low) / 2
        u_thresh = (upper_mean + q_high) / 2

        # Ensure minimum difference - safely with tensors
        thresh_diff = u_thresh - l_thresh
        if thresh_diff.item() < 0.01:
            u_thresh = l_thresh + 0.01

        # Normalize standard deviation map
        normalized_std = (std_map - l_thresh) / (u_thresh - l_thresh)
        normalized_std = torch.clamp(normalized_std, 0.0, 1.0)

        # Apply segmented sigmoid function
        mask = self._apply_segmented_sigmoid(normalized_std)

        return mask

    def _apply_segmented_sigmoid(self, norm_std):
        """Apply segmented sigmoid function for better mask distribution"""
        mask = torch.zeros_like(norm_std)

        # Define regions based on normalized standard deviation
        low_region = norm_std < 0.25
        mid_region = (norm_std >= 0.25) & (norm_std <= 0.6)
        high_region = norm_std > 0.6

        # Apply different sigmoid mappings to each region
        # For low regions (smoother curve to better differentiate subtle textures)
        mask[low_region] = torch.sigmoid(10.0 * norm_std[low_region] - 1.5) * 0.4

        # For mid regions (steeper curve for better contrast)
        mask[mid_region] = torch.sigmoid(20.0 * norm_std[mid_region] - 8.0) * 0.5 + 0.25

        # For high regions (ensure strong textures are clearly identified)
        mask[high_region] = torch.sigmoid(8.0 * norm_std[high_region] - 1.0) * 0.3 + 0.7

        return mask