import torch
from collections import OrderedDict
from copy import deepcopy
from os import path as osp

from led.archs import build_network
from led.losses import build_loss
from led.models.dual_path_model import DualPathModel
from led.utils import get_root_logger
from led.utils.registry import MODEL_REGISTRY

@MODEL_REGISTRY.register()
class DualPathGANModel(DualPathModel):
    """Dual Path Model with GAN training for RAW image denoising.

    This model extends the DualPathModel with adversarial training to improve
    perceptual quality of the output images.
    """

    def __init__(self, opt):
        super(DualPathGANModel, self).__init__(opt)

        # Initialize discriminator network
        if self.is_train:
            self.init_discriminator()

    def init_discriminator(self):
        """Initialize the discriminator network and optimizer for GAN training."""
        train_opt = self.opt['train']

        # Define discriminator network
        self.net_d = build_network(self.opt['network_d'])
        self.net_d = self.model_to_device(self.net_d)
        self.print_network(self.net_d)

        # Load pretrained discriminator if needed
        load_path = self.opt['path'].get('pretrain_network_d', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_d', 'params')
            self.load_network(
                self.net_d, load_path, self.opt['path'].get('strict_load_d', True), param_key
            )

        # Set up discriminator optimizer
        if self.opt['train'].get('optim_d'):
            optim_d_params = []
            for k, v in self.net_d.named_parameters():
                if v.requires_grad:
                    optim_d_params.append(v)
                else:
                    logger = get_root_logger()
                    logger.warning(f'Params {k} in discriminator will not be optimized.')

            optim_type = train_opt['optim_d'].pop('type')
            self.optimizer_d = self.get_optimizer(
                optim_type, optim_d_params, **train_opt['optim_d']
            )
            self.optimizers.append(self.optimizer_d)

        # Define GAN loss
        self.cri_gan = build_loss(train_opt['gan_opt']).to(self.device)
        self.gan_weight = train_opt.get('gan_weight', 1.0)

        # Additional optimizations for GAN training
        self.net_d_iters = train_opt.get('net_d_iters', 1)
        self.net_d_start_iter = train_opt.get('net_d_start_iter', 0)

    def optimize_parameters(self, current_iter):
        """Optimize model parameters for one iteration."""
        # Generate denoised output
        self.output = self.net_g(self.lq)

        # Optimize discriminator (if GAN training is active)
        if current_iter >= self.net_d_start_iter:
            # Reset gradients on discriminator
            for p in self.net_d.parameters():
                p.requires_grad = True

            # Update discriminator for multiple iterations if needed
            for _ in range(self.net_d_iters):
                self.optimizer_d.zero_grad()

                # Real detection
                spatial_real, freq_real = self.net_d(self.gt)

                # Fake detection (stop gradient to generator)
                spatial_fake, freq_fake = self.net_d(self.output.detach())

                # Discriminator loss
                d_loss = 0
                # Process all spatial scales results
                for real_pred, fake_pred in zip(spatial_real, spatial_fake):
                    d_loss_real = self.cri_gan(real_pred, True, is_disc=True)
                    d_loss_fake = self.cri_gan(fake_pred, False, is_disc=True)
                    d_loss += (d_loss_real + d_loss_fake)

                # Process frequency domain results
                d_loss_freq_real = self.cri_gan(freq_real, True, is_disc=True)
                d_loss_freq_fake = self.cri_gan(freq_fake, False, is_disc=True)
                d_loss += (d_loss_freq_real + d_loss_freq_fake)

                # Add gradient penalty if using WGAN-GP
                if self.opt['train']['gan_opt']['gan_type'] == 'wgan_gp':
                    gradient_penalty = self.cri_gan.calculate_gradient_penalty(
                        self.net_d, self.gt, self.output.detach(), self.device
                    )
                    d_loss += gradient_penalty

                d_loss.backward()
                self.optimizer_d.step()

            # Prevent discriminator from being updated in generator optimization
            for p in self.net_d.parameters():
                p.requires_grad = False

        # Optimize generator
        self.optimizer_g.zero_grad()

        # Generator losses
        l_total = 0
        loss_dict = OrderedDict()

        # Pixel loss (L1/L2)
        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.gt)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix

        # Perceptual loss (VGG)
        if self.cri_perceptual:
            l_percep = self.cri_perceptual(self.output, self.gt)
            l_total += l_percep
            loss_dict['l_percep'] = l_percep

        # Gradient/edge loss
        if self.cri_gradient:
            l_grad = self.cri_gradient(self.output, self.gt)
            l_total += l_grad
            loss_dict['l_grad'] = l_grad

        # GAN loss for generator (only after discriminator starts training)
        if current_iter >= self.net_d_start_iter:
            # Get discriminator results on generated images
            spatial_fake, freq_fake = self.net_d(self.output)

            # Calculate GAN loss for generator
            l_g_gan = 0
            # Process all spatial scales
            for fake_pred in spatial_fake:
                l_g_gan += self.cri_gan(fake_pred, True, is_disc=False)

            # Process frequency domain
            l_g_gan += self.cri_gan(freq_fake, True, is_disc=False)

            # Apply GAN weight
            l_g_gan = l_g_gan * self.gan_weight
            l_total += l_g_gan
            loss_dict['l_g_gan'] = l_g_gan

        # Backpropagate and update generator parameters
        l_total.backward()
        self.optimizer_g.step()

        # Update EMA network if needed
        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

        # Log losses
        self.log_dict = self.reduce_loss_dict(loss_dict)

    def save(self, epoch, current_iter):
        """Save networks and training states."""
        # Save generator
        if hasattr(self, 'net_g_ema'):
            self.save_network(
                [self.net_g, self.net_g_ema],
                'net_g', current_iter,
                param_key=['params', 'params_ema']
            )
        else:
            self.save_network(self.net_g, 'net_g', current_iter)

        # Save discriminator
        self.save_network(self.net_d, 'net_d', current_iter)

        # Save training state
        self.save_training_state(epoch, current_iter)