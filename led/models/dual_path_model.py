import torch
from collections import OrderedDict
from copy import deepcopy
from os import path as osp
import os

from led.archs import build_network
from led.losses import build_loss
from led.models.raw_base_model import RAWBaseModel
from led.utils import get_root_logger
from led.utils.registry import MODEL_REGISTRY
from led.losses.perceptual_loss import VGGPerceptualLoss
from led.losses.gradient_loss import GradientLoss
from led.archs.dual_path_components.noise_map import generate_noise_map

@MODEL_REGISTRY.register()
class DualPathModel(RAWBaseModel):
    """Dual Path Model for RAW image denoising.

    This model combines a detail preservation path and a noise suppression path
    to achieve both high fidelity and good visual quality.
    """

    def __init__(self, opt):
        super(DualPathModel, self).__init__(opt)

        # define network
        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        # define noise_g
        if 'noise_g' in opt:
            self.noise_g = self.build_noise_g(opt['noise_g'])
            self.noise_g = self.noise_g.to(self.device)
            self.print_network(self.noise_g)

            noise_g_path = self.opt['path'].get('predefined_noise_g', None)
            if noise_g_path is not None:
                self.load_network(self.noise_g, noise_g_path, self.opt['path'].get('strict_load_g', True), None)
            logger = get_root_logger()
            logger.info(f'Sampled Cameras: \n{self.noise_g.log_str}')

            dump_path = os.path.join(self.opt['path']['experiments_root'], 'noise_g.pth')
            torch.save(self.noise_g.state_dict(), dump_path)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), param_key)

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        # set up EMA if needed
        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define losses
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None

        # define perceptual loss if needed
        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

        # define gradient/edge loss if needed
        if train_opt.get('gradient_opt'):
            self.cri_gradient = build_loss(train_opt['gradient_opt']).to(self.device)
        else:
            self.cri_gradient = None

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def build_noise_g(self, opt):
        opt = deepcopy(opt)
        noise_g_class = eval(opt.pop('type'))
        return noise_g_class(opt, self.device)

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

    def feed_data(self, data):
        """Feed input data.

        Args:
            data (dict): Input data, should contain 'lq' and 'gt'.
        """
        self.lq = data['lq'].to(self.device)
        self.gt = data['gt'].to(self.device)
        if 'ccm' in data:
            self.ccm = data['ccm'].to(self.device)
        if 'wb' in data:
            self.wb = data['wb'].to(self.device)
        if 'ratio' in data:
            self.ratio = data['ratio'].to(self.device)
        if 'black_level' in data:
            self.black_level = data['black_level'].to(self.device)
            self.white_level = data['white_level'].to(self.device)

    def optimize_parameters(self, current_iter):
        """Optimize model parameters."""
        self.optimizer_g.zero_grad()

        noise_map = None

        # 使用噪声生成器生成训练数据（如果有）
        if hasattr(self, 'noise_g'):
            self.camera_id = torch.randint(0, len(self.noise_g), (1,)).item()
            with torch.no_grad():
                scale = self.white_level - self.black_level
                self.gt = (self.gt - self.black_level) / scale
                self.gt, self.lq, self.curr_metadata = self.noise_g(self.gt, scale, self.ratio, self.camera_id)

                # 如果网络支持噪声图，则生成噪声图
                if hasattr(self.net_g, 'use_noise_map') and self.net_g.use_noise_map:
                    noise_map = generate_noise_map(
                        image=self.lq,
                        noise_params=self.curr_metadata['noise_params'],
                        camera_params=None,  # 不需要，因为已有噪声参数
                        iso=None             # 不需要，因为已有噪声参数
                    )

                # 数据增强
                if hasattr(self, 'augment') and self.augment is not None:
                    self.gt, self.lq = self.augment(self.gt, self.lq)
                    if noise_map is not None:
                        # 对噪声图进行相同的增强
                        noise_map = self.augment(noise_map)[0]

        # 如果没有噪声生成器但需要噪声图，使用ISO计算（测试时或特定场景）
        elif hasattr(self.net_g, 'use_noise_map') and self.net_g.use_noise_map and hasattr(self, 'iso') and self.iso is not None:
            # 获取相机参数（如果有）
            camera_params = None
            camera_name = None
            if hasattr(self, 'camera_params'):
                camera_params = self.camera_params
                camera_name = getattr(self, 'camera_name', None)

            # 生成噪声图
            noise_map = generate_noise_map(
                image=self.lq,
                noise_params=None,
                camera_params=camera_params,
                iso=self.iso,
                camera_name=camera_name
            )

        # 使用噪声图（如果有）进行推理
        if noise_map is not None and hasattr(self.net_g, 'use_noise_map') and self.net_g.use_noise_map:
            self.output = self.net_g(self.lq, noise_map)
        else:
            self.output = self.net_g(self.lq)

        l_total = 0
        loss_dict = OrderedDict()

        # pixel loss
        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.gt)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix

        # perceptual loss
        if self.cri_perceptual:
            l_percep = self.cri_perceptual(self.output, self.gt)
            l_total += l_percep
            loss_dict['l_percep'] = l_percep

        # gradient/edge loss
        if self.cri_gradient:
            l_grad = self.cri_gradient(self.output, self.gt)
            l_total += l_grad
            loss_dict['l_grad'] = l_grad

        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def test(self):
        """Forward function used in testing."""
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output = self.net_g_ema(self.lq)
                if self.corrector is not None:
                    self.output = self.corrector(self.output, self.gt)
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.output = self.net_g(self.lq)
                if self.corrector is not None:
                    self.output = self.corrector(self.output, self.gt)
            self.net_g.train()

    def save(self, epoch, current_iter):
        """Save networks and training states."""
        if hasattr(self, 'net_g_ema'):
            self.save_network(
                [self.net_g, self.net_g_ema],
                'net_g', current_iter,
                param_key=['params', 'params_ema']
            )
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)