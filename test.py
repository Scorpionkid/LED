import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from basicsr.utils.registry import ARCH_REGISTRY

class FrequencyDomainHead(nn.Module):
    """频域判别头，处理频域特征"""
    def __init__(self, in_channels=4):
        super(FrequencyDomainHead, self).__init__()

        # 卷积层处理频谱图
        self.conv1 = nn.Conv2d(2, 32, 3, stride=1, padding=1)  # 幅度和相位
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(128, 1, 3, stride=1, padding=1)

        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        # 确保输入是正确的形状
        x_freq = self._fft_shift(x)

        # 分离幅度和相位
        magnitude = torch.abs(x_freq)
        phase = torch.angle(x_freq)

        # 将幅度和相位结合为输入
        x_combined = torch.cat([magnitude, phase], dim=1)

        # 前向传播
        x = self.leaky_relu(self.conv1(x_combined))
        x = self.leaky_relu(self.conv2(x))
        x = self.leaky_relu(self.conv3(x))
        x = self.conv4(x)

        return x

    def _fft_shift(self, x):
        """对输入应用FFT并移动低频到中心"""
        # 处理多通道输入
        if x.size(1) > 1:
            # 处理RGGB格式
            if x.size(1) == 4:  # RAW格式
                # 转换为灰度图
                x = 0.299 * x[:,0:1] + 0.587 * (x[:,1:2] + x[:,2:3])/2 + 0.114 * x[:,3:4]
            else:  # RGB格式
                # 转换为灰度图
                x = 0.299 * x[:,0:1] + 0.587 * x[:,1:2] + 0.114 * x[:,2:3]

        # 执行2D FFT
        x_freq = torch.fft.fft2(x)

        # 移动低频到中心
        x_freq = torch.fft.fftshift(x_freq)

        return x_freq

@ARCH_REGISTRY.register()
class MultiScaleVGGFreqDiscriminator(nn.Module):
    """多尺度VGG特征判别器与频域鉴别头相结合的判别器"""
    def __init__(self, in_channels=4, use_pretrained_vgg=True):
        super(MultiScaleVGGFreqDiscriminator, self).__init__()

        # 加载预训练VGG特征提取器
        if use_pretrained_vgg:
            vgg = models.vgg19(pretrained=True).features
        else:
            vgg = models.vgg19(pretrained=False).features

        self.vgg_blocks = nn.ModuleList([
            nn.Sequential(*list(vgg.children())[:4]),   # 低级特征
            nn.Sequential(*list(vgg.children())[4:9]),  # 中级特征
            nn.Sequential(*list(vgg.children())[9:18])  # 高级特征
        ])

        # 预处理层 - 将RAW转换为RGB格式，适应VGG输入
        self.preprocess = nn.Conv2d(in_channels, 3, 1)

        # 多尺度判别头
        self.spatial_heads = nn.ModuleList([
            self._make_head(64),   # 低级特征判别头
            self._make_head(128),  # 中级特征判别头
            self._make_head(256)   # 高级特征判别头
        ])

        # 频域判别头
        self.freq_head = FrequencyDomainHead(in_channels)

        # 如果使用预训练VGG，冻结VGG参数
        if use_pretrained_vgg:
            for param in self.vgg_blocks.parameters():
                param.requires_grad = False

        # 注册均值和标准差
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _make_head(self, in_channels):
        """创建判别头"""
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels//2, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels//2, in_channels//4, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels//4, 1, 3, stride=1, padding=1)
        )

    def forward(self, x):
        # 预处理RAW图像为RGB格式
        x_rgb = self.preprocess(x)

        # 规范化输入
        x_norm = (x_rgb - self.mean) / self.std

        # 多尺度VGG特征提取
        feat_maps = []
        feat = x_norm
        for block in self.vgg_blocks:
            feat = block(feat)
            feat_maps.append(feat)

        # 空间域判别结果
        spatial_results = []
        for feat_map, head in zip(feat_maps, self.spatial_heads):
            spatial_results.append(head(feat_map))

        # 频域判别结果
        freq_result = self.freq_head(x)

        return spatial_results, freq_result