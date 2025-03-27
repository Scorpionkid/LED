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
        """前向传播

        Args:
            pred (Tensor): 预测值 [B, C, H, W]
            target (Tensor): 目标值 [B, C, H, W]
            texture_mask (Tensor, optional): 纹理掩码 [B, 1, H, W]，值范围[0,1]
            weight (Tensor, optional): 额外的权重掩码

        Returns:
            loss (Tensor): 计算的损失值
        """
        # 如果没有提供纹理掩码，使用固定权重
        if texture_mask is None:
            return self.pixel_loss_fn(pred, target, weight, reduction=self.reduction)

        # 确保纹理掩码与输入尺寸匹配
        if texture_mask.shape[2:] != pred.shape[2:]:
            texture_mask = F.interpolate(texture_mask, size=pred.shape[2:], mode='bilinear')

        # 计算平坦区域和纹理区域的掩码
        flat_mask = 1.0 - texture_mask
        texture_mask_binary = (texture_mask > self.texture_threshold).float()
        flat_mask_binary = 1.0 - texture_mask_binary

        # 对纹理和平坦区域计算像素损失
        flat_loss = self.pixel_loss_fn(pred, target, flat_mask * weight if weight is not None else flat_mask, reduction='sum')
        texture_loss = self.pixel_loss_fn(pred, target, texture_mask * weight if weight is not None else texture_mask, reduction='sum')

        # 计算归一化权重
        flat_area = torch.sum(flat_mask) + 1e-8
        texture_area = torch.sum(texture_mask) + 1e-8

        # 归一化损失
        flat_loss = flat_loss / flat_area
        texture_loss = texture_loss / texture_area

        # 计算最终的像素损失
        pixel_loss = self.flat_weight * flat_loss + self.texture_weight * texture_loss

        # 初始化总损失
        total_loss = pixel_loss

        # 添加纹理区域的感知损失
        if self.perceptual_loss is not None and self.perceptual_weight > 0 and texture_area > 1e-6:
            # 只在有足够的纹理区域时计算感知损失
            if torch.sum(texture_mask_binary) > 0:
                # 使用二进制掩码，仅保留纹理区域的信息
                pred_texture = pred * texture_mask_binary
                target_texture = target * texture_mask_binary
                perceptual_loss = self.perceptual_loss(pred_texture, target_texture)
                total_loss = total_loss + self.perceptual_weight * perceptual_loss

        # 添加纹理区域的梯度损失
        if self.gradient_loss is not None and self.gradient_weight > 0 and texture_area > 1e-6:
            # 只在有足够的纹理区域时计算梯度损失
            if torch.sum(texture_mask_binary) > 0:
                pred_texture = pred * texture_mask_binary
                target_texture = target * texture_mask_binary
                gradient_loss = self.gradient_loss(pred_texture, target_texture)
                total_loss = total_loss + self.gradient_weight * gradient_loss

        # 动态调整权重（训练时根据损失自动平衡）
        if self.training and self.use_dynamic_weights:
            # 跟踪并更新平坦和纹理区域的权重（使用指数移动平均）
            if not hasattr(self, 'flat_loss_ema'):
                self.flat_loss_ema = flat_loss.detach()
                self.texture_loss_ema = texture_loss.detach()
            else:
                self.flat_loss_ema = 0.9 * self.flat_loss_ema + 0.1 * flat_loss.detach()
                self.texture_loss_ema = 0.9 * self.texture_loss_ema + 0.1 * texture_loss.detach()

            # 计算动态权重
            ratio = self.flat_loss_ema / (self.texture_loss_ema + 1e-8)
            self.flat_weight = 1.0
            self.texture_weight = torch.clamp(ratio, 0.5, 2.0).item()  # 限制范围，防止过度调整

        return total_loss