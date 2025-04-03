import torch
import torch.nn as nn
import torch.nn.functional as F

from led.utils.registry import LOSS_REGISTRY
from led.losses.loss_util import weighted_loss
from led.losses.basic_loss import l1_loss, mse_loss
from led.losses.perceptual_loss import VGGPerceptualLoss
from led.losses.gradient_loss import GradientLoss  # 使用已有的GradientLoss

@LOSS_REGISTRY.register()
class AdaptiveTextureLoss(nn.Module):
    """基于纹理的自适应损失函数

    对纹理区域和平坦区域应用不同权重和类型的损失

    Args:
        loss_weight()
        pixel_loss_type (str): 像素损失类型，'l1'或'l2'
        flat_weight (float): 平坦区域的损失权重
        texture_weight (float): 纹理区域的损失权重
        perceptual_weight (float): 感知损失权重，仅用于纹理区域
        gradient_weight (float): 梯度损失权重，仅用于纹理区域
        texture_threshold (float): 判断纹理区域的阈值，>threshold视为纹理区域
        reduction (str): 损失归约方式
    """
    def __init__(self, pixel_loss_type='l1', flat_weight=1.0, texture_weight=1.0,
                 perceptual_weight=0.1, gradient_weight=0.1, texture_threshold=0.5,
                 use_dynamic_weights=True, reduction='mean', loss_weight=1.0):
        super(AdaptiveTextureLoss, self).__init__()
        self.flat_weight = flat_weight * loss_weight
        self.texture_weight = texture_weight * loss_weight
        self.perceptual_weight = perceptual_weight * loss_weight
        self.gradient_weight = gradient_weight * loss_weight
        self.texture_threshold = texture_threshold
        self.reduction = reduction
        self.use_dynamic_weights = use_dynamic_weights

        # 选择像素损失类型
        if pixel_loss_type.lower() == 'l1':
            self.pixel_loss_fn = l1_loss
        elif pixel_loss_type.lower() == 'l2' or pixel_loss_type.lower() == 'mse':
            self.pixel_loss_fn = mse_loss
        else:
            raise ValueError(f"不支持的像素损失类型: {pixel_loss_type}")

        # 感知损失 - 只用于纹理区域
        if perceptual_weight > 0:
            self.perceptual_loss = VGGPerceptualLoss(
                layer_weights={'conv3_4': 0.25, 'conv4_4': 0.5, 'conv5_4': 0.25},
                normalize=True
            )
        else:
            self.perceptual_loss = None

        # 梯度损失 - 只用于纹理区域
        if gradient_weight > 0:
            # 使用已注册的GradientLoss
            self.gradient_loss = GradientLoss(loss_weight=1.0)
        else:
            self.gradient_loss = None

    def forward(self, pred, target, texture_mask=None, weight=None, **kwargs):
        # 如果没有纹理掩码，使用基本损失
        if texture_mask is None:
            return self.pixel_loss_fn(pred, target, weight, reduction=self.reduction)

        # 确保纹理掩码尺寸匹配
        if texture_mask.shape[2:] != pred.shape[2:]:
            texture_mask = F.interpolate(texture_mask, size=pred.shape[2:], mode='bilinear')

        # 计算平坦区域掩码
        flat_mask = 1.0 - texture_mask

        # 计算区域比例（更安全的方式）
        total_pixels = float(pred.shape[2] * pred.shape[3] * pred.shape[0])
        texture_ratio = torch.sum(texture_mask) / total_pixels
        flat_ratio = 1.0 - texture_ratio

        # 确保比例在合理范围内
        texture_ratio = torch.clamp(texture_ratio, 0.05, 0.95)
        flat_ratio = torch.clamp(flat_ratio, 0.05, 0.95)

        # 对纹理和平坦区域使用mean而非sum，避免归一化问题
        flat_loss = self.pixel_loss_fn(pred, target, flat_mask, reduction='mean')
        texture_loss = self.pixel_loss_fn(pred, target, texture_mask, reduction='mean')

        # 初始静态权重
        flat_weight = self.flat_weight
        texture_weight = self.texture_weight

        # 动态权重调整（更稳定的实现）
        if self.training and self.use_dynamic_weights:
            # 使用log-space计算比值，避免除法带来的不稳定性
            with torch.no_grad():  # 确保不影响梯度计算
                # 计算losses的对数比
                if not hasattr(self, 'loss_ratio_ema'):
                    self.loss_ratio_ema = torch.tensor(0.0, device=pred.device)

                # 直接计算对数比，避免除法
                log_ratio = torch.log(flat_loss + 1e-6) - torch.log(texture_loss + 1e-6)

                # 限制单步变化幅度，避免剧烈变化
                log_ratio = torch.clamp(log_ratio, -0.5, 0.5)

                # 更新EMA
                self.loss_ratio_ema = 0.95 * self.loss_ratio_ema + 0.05 * log_ratio

                # 将对数比转换为权重调整因子
                adjust_factor = torch.exp(self.loss_ratio_ema)
                adjust_factor = torch.clamp(adjust_factor, 0.8, 1.25)  # 更温和的调整范围

                # 应用调整
                texture_weight = self.texture_weight * adjust_factor.item()
                # 保持flat_weight不变，只调整texture_weight

        pixel_loss = flat_weight * flat_loss + texture_weight * texture_loss

        # 使用区域比例加权，而非直接归一化
        pixel_loss = self.flat_weight * flat_loss + self.texture_weight * texture_loss

        # 初始化总损失
        total_loss = pixel_loss

        # 仅当纹理区域足够大时添加感知和梯度损失
        if texture_ratio > 0.1:
            if self.perceptual_loss is not None and self.perceptual_weight > 0:
                # 使用软掩码而非二元掩码
                pred_texture = pred * texture_mask
                target_texture = target * texture_mask
                perceptual_loss = self.perceptual_loss(pred_texture, target_texture)
                total_loss += self.perceptual_weight * perceptual_loss

            if self.gradient_loss is not None and self.gradient_weight > 0:
                gradient_loss = self.gradient_loss(pred * texture_mask, target * texture_mask)
                total_loss += self.gradient_weight * gradient_loss

        # 检查损失值稳定性
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            return self.pixel_loss_fn(pred, target, reduction='mean')

        return total_loss