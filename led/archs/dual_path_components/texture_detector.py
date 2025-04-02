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
    def __init__(self, window_sizes=[11, 25, 49], adaptive_thresh=True, raw_channels=4,
                 noise_sensitivity=3.0):
        super(RAWTextureDetector, self).__init__()
        self.window_sizes = window_sizes
        self.adaptive_thresh = adaptive_thresh
        self.raw_channels = raw_channels
        # self.noise_sensitivity = nn.Parameter(torch.tensor(noise_sensitivity))

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
        lower_thresh, upper_thresh = self._compute_adaptive_thresholds(x, fused_std_map)

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
        # result = result / (torch.mean(result) + 1e-6)

        return result

    def _compute_adaptive_thresholds(self, x, std_map):
        """计算完全基于图像统计量的自适应阈值"""

        # 计算std_map的分位点
        std_flat = std_map.view(-1)
        q25 = torch.quantile(std_flat, 0.25)
        q75 = torch.quantile(std_flat, 0.75)

        # 计算四分位距(IQR)
        iqr = q75 - q25

        # 基于全局标准差
        global_std = torch.std(x, dim=(2, 3), keepdim=True)

        # 直接使用分位点计算基础阈值，而非使用预设的base_lower/upper_thresh
        lower_thresh = (q25 - 0.5 * iqr).view(1, 1, 1, 1).expand_as(global_std)
        upper_thresh = (q75 + 0.5 * iqr).view(1, 1, 1, 1).expand_as(global_std)

        # 应用全局标准差调整
        global_factor = torch.clamp(global_std * 5.0, 0.5, 2.0)
        lower_thresh = lower_thresh * global_factor
        upper_thresh = upper_thresh * global_factor

        # 应用在通道维度上的权重(保留原有功能)
        channel_weights = F.softmax(self.channel_weights, dim=0).view(1, self.raw_channels, 1, 1)
        weighted_x = x * channel_weights
        channel_std = torch.std(weighted_x, dim=(2, 3), keepdim=True)

        # 进一步基于各通道的特性调整阈值(保留原有功能)
        channel_factor = torch.clamp(channel_std * 2.0, 0.8, 1.2)
        lower_thresh = lower_thresh * channel_factor
        upper_thresh = upper_thresh * channel_factor

        return lower_thresh, upper_thresh

    def _generate_texture_mask(self, std_map, lower_thresh, upper_thresh):
        # 获取std_map的基本统计量
        std_flat = std_map.view(-1)
        q30 = torch.quantile(std_flat, 0.30)
        q70 = torch.quantile(std_flat, 0.70)

        # 将传入的lower_thresh和upper_thresh转换为单一值
        lower_mean = torch.mean(lower_thresh, dim=1, keepdim=True)  # 例如：0.0076
        upper_mean = torch.mean(upper_thresh, dim=1, keepdim=True)  # 例如：0.0758

        # 结合传入阈值和实际分布情况
        l_thresh = 0.3 * lower_mean + 0.7 * q30  # 例如：~0.0302
        u_thresh = 0.3 * upper_mean + 0.7 * q70  # 例如：~0.0473

        # 检查分布的集中程度
        iqr = q70 - q30  # 例如：0.0070

        # 如果分布过于集中，扩展阈值范围
        if iqr < 0.01:  # 会触发此条件
            # 扩展阈值，但仍参考原始阈值
            l_thresh = l_thresh - 0.5 * iqr  # 例如：~0.0267
            u_thresh = u_thresh + 0.5 * iqr  # 例如：~0.0508

        # 确保最小阈值差距
        thresh_diff = u_thresh - l_thresh
        if thresh_diff < 0.015:  # 增加最小差距
            midpoint = (l_thresh + u_thresh) / 2
            l_thresh = midpoint - 0.0075
            u_thresh = midpoint + 0.0075

        # 计算sigmoid参数
        midpoint = (l_thresh + u_thresh) / 2  # 例如：~0.0387
        scale = u_thresh - l_thresh  # 例如：~0.0241

        # 使用sigmoid进行平滑非线性映射
        steepness = 8.0
        normalized = torch.sigmoid(steepness * (std_map - midpoint) / scale)

        # 缩放到[0.1, 0.8]范围，避免极端值
        mask = 0.1 + 0.7 * normalized

        return mask

    def _apply_segmented_sigmoid(self, norm_std):
        """Apply segmented sigmoid function for better mask distribution"""
        mask = torch.zeros_like(norm_std)

        # Define regions based on normalized standard deviation
        low_region = norm_std < 0.3
        mid_region = (norm_std >= 0.3) & (norm_std <= 0.7)
        high_region = norm_std > 0.7

        # Apply different sigmoid mappings to each region
        # For low regions (smoother curve to better differentiate subtle textures)
        # mask[low_region] = torch.sigmoid(8 * norm_std[low_region] - 1.5) * 0.3

        # For mid regions (steeper curve for better contrast)
        # mask[mid_region] = torch.sigmoid(20.0 * norm_std[mid_region] - 8.0) * 0.5 + 0.25

        # For high regions (ensure strong textures are clearly identified)
        # mask[high_region] = torch.sigmoid(8.0 * norm_std[high_region] - 1.0) * 0.3 + 0.7

        # 降低整体输出范围，避免掩码过饱和
        # 低区域：最大输出从0.4降到0.3
        mask[low_region] = torch.sigmoid(8.0 * norm_std[low_region] - 1.5) * 0.3

        # 中间区域：范围从[0.25,0.75]降到[0.3,0.6]
        mask[mid_region] = torch.sigmoid(12.0 * norm_std[mid_region] - 6.0) * 0.3 + 0.3

        # 高区域：最大输出从1.0降到0.8
        mask[high_region] = torch.sigmoid(6.0 * norm_std[high_region] - 1.5) * 0.2 + 0.6

        return mask