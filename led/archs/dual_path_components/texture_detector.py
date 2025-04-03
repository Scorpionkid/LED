import torch
import torch.nn as nn
import torch.nn.functional as F
from led.utils.logger import get_root_logger

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
        # self.noise_sensitivity = nn.Parameter(torch.tensor(noise_sensitivity))
        self.logger = get_root_logger()

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

        # 在return texture_mask之前添加
        if hasattr(self, 'debug_iter') and self.debug_iter % 100 == 0:
            import os
            import matplotlib.pyplot as plt
            import numpy as np

            os.makedirs("debug/complete_debug", exist_ok=True)

            # 创建包含更多信息的综合可视化
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))

            # 原始图像显示
            if x.size(1) == 4:  # RAW图像RGGB格式
                # 简化显示RAW图像
                display_img = 0.299 * x[0,0:1].detach().cpu() + 0.587 * (x[0,1:2].detach().cpu() + x[0,2:3].detach().cpu())/2 + 0.114 * x[0,3:4].detach().cpu()
                display_img = display_img.squeeze().numpy()
            else:
                display_img = torch.mean(x[0].detach().cpu(), dim=0).numpy()

            axes[0, 0].imshow(display_img, cmap='gray')
            axes[0, 0].set_title('输入图像(灰度表示)')

            # 特征图
            feat_vis = torch.mean(features[0].detach().cpu(), dim=0).numpy()
            axes[0, 1].imshow(feat_vis, cmap='magma')
            axes[0, 1].set_title('特征图(通道平均)')

            # 融合后的标准差图
            axes[1, 0].imshow(fused_std_map[0, 0].detach().cpu().numpy(), cmap='plasma')
            axes[1, 0].set_title('融合标准差图')

            # 最终纹理掩码
            mask_img = axes[1, 1].imshow(texture_mask[0, 0].detach().cpu().numpy(), cmap='viridis')
            axes[1, 1].set_title('最终纹理掩码')
            fig.colorbar(mask_img, ax=axes[1, 1])

            # 添加统计信息
            plt.figtext(0.02, 0.01,
                    f"掩码统计: 最小值={torch.min(texture_mask).item():.4f}, 最大值={torch.max(texture_mask).item():.4f}, "
                    f"平均值={torch.mean(texture_mask).item():.4f}, 中位数={torch.median(texture_mask.view(-1)).item():.4f}",
                    fontsize=12, bbox=dict(facecolor='white', alpha=0.8))

            plt.tight_layout()
            plt.savefig(f"debug/complete_debug/complete_vis_{self.debug_iter}.png", dpi=150)
            plt.close()

        return texture_mask

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

            if torch.isnan(local_var).any():
                self.logger.warning(f"局部方差图有nan")

            result += channel_weights[c] * torch.sqrt(local_var)

        result = torch.pow(result, 0.8)
        # result = result / (torch.mean(result) + 1e-6)

        # 添加可视化代码
        if not hasattr(self, 'debug_iter'):
            self.debug_iter = 0
        else:
            self.debug_iter += 1

        if self.debug_iter % 100 == 0:  # 每100次迭代保存一次
            import matplotlib.pyplot as plt
            import os
            os.makedirs("debug/std_maps", exist_ok=True)
            plt.figure(figsize=(10, 8))
            plt.imshow(result[0, 0].detach().cpu().numpy(), cmap='viridis')
            plt.colorbar(label='标准差值')
            plt.title(f'窗口大小:{window_size} 标准差图')
            plt.savefig(f"debug/std_maps/std_map_w{window_size}_{self.debug_iter}.png")
            plt.close()

        return result

    def _compute_adaptive_thresholds(self, x, std_map):
        """计算完全基于图像统计量的自适应阈值"""

        if torch.isnan(std_map).any():
            self.logger.warning(f"_compute_adaptive_thresholds的std_map: nan")

        # 计算std_map的分位点
        std_flat = std_map.view(-1)
        std_flat = std_flat[~torch.isnan(std_flat)]
        if std_flat.numel() == 0:
            return torch.tensor(self.base_lower_thresh, device=x.device), torch.tensor(self.base_upper_thresh, device=x.device)

        q25 = torch.quantile(std_flat, 0.25)
        q75 = torch.quantile(std_flat, 0.75)

        # 计算四分位距(IQR)
        iqr = q75 - q25

        if iqr < 1e-5:
            iqr = torch.tensor(0.05, device=x.device)

        # 基于全局标准差
        global_std = torch.std(x, dim=(2, 3), keepdim=True)

        # # 直接使用分位点计算基础阈值，而非使用预设的base_lower/upper_thresh
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

         # 确保阈值不会导致除零错误
        if torch.abs(upper_thresh - lower_thresh).min() < 1e-5:
            mean_thresh = (upper_thresh + lower_thresh) / 2
            lower_thresh = mean_thresh - 0.05
            upper_thresh = mean_thresh + 0.05

        print(f"std_map 统计: min={torch.min(std_map).item()}, max={torch.max(std_map).item()}, has_nan={torch.isnan(std_map).any().item()}")
        print(f"阈值: lower={torch.min(lower_thresh).item()}, upper={torch.max(upper_thresh).item()}, diff={torch.min(upper_thresh-lower_thresh).item()}")

        return lower_thresh, upper_thresh

    def _generate_texture_mask(self, std_map, lower_thresh, upper_thresh):
        if torch.isnan(std_map).any():
            self.logger.warning(f"_generate_texture_mask的std_map: nan")

        # 安全的归一化操作
        diff = torch.clamp(upper_thresh - lower_thresh, min=1e-5)
        normalized_std = (std_map - lower_thresh) / diff
        normalized_std = torch.clamp(normalized_std, 0.0, 1.0)

        if torch.isnan(normalized_std).any():
            self.logger.warning(f"_generate_texture_mask生成的normalized_std有nan")


        # 使用sigmoid函数实现平滑过渡
        texture_mask = torch.sigmoid(6.0 * normalized_std - 3.0)  # 调整斜率使过渡更加明显
        texture_mask = torch.mean(texture_mask, dim=1, keepdim=True)

        if torch.isnan(texture_mask).any():
            # texture_mask = torch.nan_to_num(texture_mask, nan=0.5)
            self.logger.warning(f"_generate_texture_mask生成的掩码有nan")

        # 添加可视化代码
        if hasattr(self, 'debug_iter') and self.debug_iter % 100 == 0:
            import matplotlib.pyplot as plt
            import os
            import numpy as np
            os.makedirs("debug/texture_masks", exist_ok=True)

            # 保存标准差和纹理掩码的对比图
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))

            # 原始标准差图
            im0 = axes[0].imshow(std_map[0, 0].detach().cpu().numpy(), cmap='plasma')
            axes[0].set_title('原始标准差图')
            fig.colorbar(im0, ax=axes[0])

            # 归一化后的标准差图
            im1 = axes[1].imshow(normalized_std[0, 0].detach().cpu().numpy(), cmap='plasma')
            axes[1].set_title('归一化标准差')
            fig.colorbar(im1, ax=axes[1])

            # 最终纹理掩码
            im2 = axes[2].imshow(texture_mask[0, 0].detach().cpu().numpy(), cmap='viridis')
            axes[2].set_title('最终纹理掩码')
            fig.colorbar(im2, ax=axes[2])

            plt.tight_layout()
            plt.savefig(f"debug/texture_masks/mask_generation_{self.debug_iter}.png")
            plt.close()

        return texture_mask