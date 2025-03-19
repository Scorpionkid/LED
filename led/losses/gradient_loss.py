import torch
import torch.nn as nn
import torch.nn.functional as F
from led.utils.registry import LOSS_REGISTRY

@LOSS_REGISTRY.register()
class GradientLoss(nn.Module):
    """Gradient/edge-preserving loss"""
    def __init__(self, loss_weight=1.0):
        super(GradientLoss, self).__init__()
        self.loss_weight = loss_weight

        # create Sobel operator
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                              dtype=torch.float32).reshape(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                              dtype=torch.float32).reshape(1, 1, 3, 3)

        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def forward(self, pred, target):
        # ensure the input is in the correct shape
        b, c, h, w = pred.shape
        pred_gray = pred.mean(dim=1, keepdim=True)
        target_gray = target.mean(dim=1, keepdim=True)

        # apply Sobel operator
        grad_x_pred = F.conv2d(pred_gray, self.sobel_x, padding=1)
        grad_y_pred = F.conv2d(pred_gray, self.sobel_y, padding=1)
        grad_x_target = F.conv2d(target_gray, self.sobel_x, padding=1)
        grad_y_target = F.conv2d(target_gray, self.sobel_y, padding=1)

        # calculate L1 loss
        loss = F.l1_loss(grad_x_pred, grad_x_target) + F.l1_loss(grad_y_pred, grad_y_target)

        return loss * self.loss_weight