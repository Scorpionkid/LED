import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.utils.registry import LOSS_REGISTRY

@LOSS_REGISTRY.register()
class GANLoss(nn.Module):
    """GAN损失函数"""
    def __init__(self, gan_type='vanilla', real_label_val=1.0, fake_label_val=0.0, loss_weight=1.0):
        super(GANLoss, self).__init__()
        self.gan_type = gan_type
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val
        self.loss_weight = loss_weight

        if gan_type == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_type == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_type == 'wgan':
            def wgan_loss(input, target):
                # target is boolean
                return -input.mean() if target else input.mean()
            self.loss = wgan_loss
        elif gan_type == 'wgan_gp':
            def wgan_loss(input, target):
                # target is boolean
                return -input.mean() if target else input.mean()
            self.loss = wgan_loss
        else:
            raise NotImplementedError(f'GAN类型 {gan_type} 未实现.')

    def forward(self, input, target_is_real):
        """
        Args:
            input (Tensor): 输入张量
            target_is_real (bool): 目标是否为真实图像
        """
        target_label = target_is_real * self.real_label_val + (1 - target_is_real) * self.fake_label_val

        if self.gan_type == 'vanilla' or self.gan_type == 'lsgan':
            target_tensor = torch.ones_like(input) * target_label
            loss = self.loss(input, target_tensor)
        elif 'wgan' in self.gan_type:
            loss = self.loss(input, target_is_real)

        return loss * self.loss_weight

    def calculate_gradient_penalty(self, netD, real_data, fake_data, device):
        """计算WGAN-GP梯度惩罚"""
        if self.gan_type != 'wgan_gp':
            return 0.0

        batch_size = real_data.size(0)

        # 随机插值系数
        alpha = torch.rand(batch_size, 1, 1, 1, device=device)

        # 在真实和生成数据之间进行线性插值
        interpolates = alpha * real_data + (1 - alpha) * fake_data
        interpolates.requires_grad_(True)

        # 通过判别器
        spatial_preds, freq_pred = netD(interpolates)

        # 计算梯度
        gradients = []
        for pred in spatial_preds:
            grad = torch.autograd.grad(
                outputs=pred.sum(), inputs=interpolates,
                create_graph=True, retain_graph=True)[0]
            gradients.append(grad)

        grad_freq = torch.autograd.grad(
            outputs=freq_pred.sum(), inputs=interpolates,
            create_graph=True, retain_graph=True)[0]
        gradients.append(grad_freq)

        # 计算梯度惩罚
        gradient_penalty = 0.0
        for grad in gradients:
            grad = grad.view(batch_size, -1)
            grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
            gradient_penalty += torch.mean((grad_l2norm - 1) ** 2)

        # 平均所有梯度惩罚
        gradient_penalty = gradient_penalty / len(gradients)

        return gradient_penalty * 10.0  # lambda=10