import torch
from collections import OrderedDict
from copy import deepcopy
from os import path as osp
import os
from torch.cuda.amp import autocast, GradScaler

from led.archs import build_network
from led.losses import build_loss
from led.models.raw_base_model import RAWBaseModel
from led.utils import get_root_logger
from led.utils.registry import MODEL_REGISTRY
from led.losses.perceptual_loss import VGGPerceptualLoss
from led.losses.gradient_loss import GradientLoss
from led.losses.adaptive_texture_loss import AdaptiveTextureLoss
from led.models.value_monitor import ValueMonitor

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
        self.use_noise_map = opt['network_g'].get('use_noise_map', False)
        self.use_texture_detection = opt['network_g'].get('use_texture_detection', False)
        logger = get_root_logger()
        logger.info(f"Configured use_noise_map: {self.use_noise_map}")
        logger.info(f"Configured texture_detector: {self.use_texture_detection}")

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

        # camera_params
        if 'camera_params' in opt:
            # 直接提供的相机参数
            self.camera_params = opt['camera_params']
            self.camera_name = opt.get('camera_name', None)

        elif 'noise_g' in opt:
            # 从噪声生成器获取相机参数
            self.noise_g = self.build_noise_g(opt['noise_g'])

            # 检查噪声生成器类型，对不同类型进行不同处理
            if hasattr(self.noise_g, 'camera_params'):
                # CalibratedNoisyPairGenerator
                self.camera_params = self.noise_g.camera_params
                if len(self.camera_params) == 1:
                    self.camera_name = list(self.camera_params.keys())[0]
                else:
                    self.camera_name = opt.get('camera_name', None)

            elif hasattr(self.noise_g, 'virtual_camera_count'):
                # VirtualNoisyPairGenerator
                self.camera_params = self.noise_g.json_dict
                # 默认使用第一个虚拟相机
                #TODO使用实际加噪相机
                self.camera_name = opt.get('camera_name', 'IC0')



        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), param_key)

        if self.is_train:
            self.init_training_settings()
            self.init_monitoring()

        # Add support for mixed precision training.
        self.use_amp = opt.get('use_amp', False)
        if self.use_amp:
            self.scaler = GradScaler()
            print("Using Automatic Mixed Precision training")

        # gradient accumulation
        # self.accumulation_steps = opt.get('accumulation_steps', 1)
        # if self.accumulation_steps > 1:
        #     print(f"Using gradient accumulation: {self.accumulation_steps} steps")

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
            if self.use_texture_detection and train_opt.get('adaptive_loss', True):
                train_opt['pixel_opt']['type'] = 'AdaptiveTextureLoss'
                self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)

                self.cri_perceptual = None
                self.cri_gradient = None

                logger = get_root_logger()
                logger.info("Using AdaptiveTextureLoss with integrated perceptual and gradient losses")
                logger.info("Global perceptual and gradient losses are disabled to avoid duplication")
            else:
                self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)

                if train_opt.get('perceptual_opt'):
                    self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
                else:
                    self.cri_perceptual = None

                if train_opt.get('gradient_opt'):
                    self.cri_gradient = build_loss(train_opt['gradient_opt']).to(self.device)
                else:
                    self.cri_gradient = None
        else:
            self.cri_pix = None
            self.cri_perceptual = None
            self.cri_gradient = None

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def init_monitoring(self):
        """初始化模型监控系统"""
        self.value_monitor = ValueMonitor()
        self.value_monitor.register_hooks(self.net_g, prefix='net_g')

        # 为锐度恢复模块添加特殊钩子（重点监控）
        if hasattr(self.net_g, 'sharpness_recovery'):
            def detailed_sharpness_hook(module, inputs, outputs):
                if not hasattr(self, 'monitor_iter'):
                    self.monitor_iter = 0
                self.monitor_iter += 1

                if self.monitor_iter % 10 == 0:  # 每10次迭代记录一次，避免过多日志
                    with torch.no_grad():
                        # 检查输入
                        if isinstance(inputs, tuple) and len(inputs) > 0:
                            input_tensor = inputs[0]
                            input_stats = {
                                'min': torch.min(input_tensor).item(),
                                'max': torch.max(input_tensor).item(),
                                'mean': torch.mean(input_tensor).item(),
                                'std': torch.std(input_tensor).item(),
                                'has_nan': torch.isnan(input_tensor).any().item(),
                                'has_inf': torch.isinf(input_tensor).any().item()
                            }

                        # 检查输出
                        output_stats = {
                            'min': torch.min(outputs).item(),
                            'max': torch.max(outputs).item(),
                            'mean': torch.mean(outputs).item(),
                            'std': torch.std(outputs).item(),
                            'has_nan': torch.isnan(outputs).any().item(),
                            'has_inf': torch.isinf(outputs).any().item()
                        }

                        # 检查是否有异常值
                        if (output_stats['max'] > 2.0 or output_stats['has_nan']
                            or output_stats['has_inf'] or input_stats['has_nan']
                            or input_stats['has_inf']):

                            print(f"\n[警告] 锐度恢复模块检测到异常值 (iter {self.monitor_iter}):")
                            print(f"输入统计: {input_stats}")
                            print(f"输出统计: {output_stats}")

                            # 保存详细信息用于后续分析
                            os.makedirs("debug/abnormal_values", exist_ok=True)
                            torch.save({
                                'iter': self.monitor_iter,
                                'input': input_tensor.detach().cpu(),
                                'output': outputs.detach().cpu(),
                                'input_stats': input_stats,
                                'output_stats': output_stats,
                                'texture_mask': inputs[2].detach().cpu() if len(inputs) > 2 else None
                            }, f"debug/abnormal_values/sharpness_abnormal_{self.monitor_iter}.pth")

            # 注册详细的锐度恢复模块钩子
            self.sharpness_hook = self.net_g.sharpness_recovery.register_forward_hook(
                detailed_sharpness_hook
            )

        # 创建调试目录
        os.makedirs("debug", exist_ok=True)
        print("模型监控系统已初始化，将监控所有模块的异常值")

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
        if 'iso' in data:
            self.iso = data['iso'].to(self.device)
        if 'black_level' in data:
            self.black_level = data['black_level'].to(self.device)
            self.white_level = data['white_level'].to(self.device)


    def optimize_parameters(self, current_iter):
        """Optimize model parameters."""
        #TODO
        # if current_iter % self.accumulation_steps == 0:
        #     self.optimizer_g.zero_grad()

        if hasattr(self, 'value_monitor'):
            self.value_monitor.update_iter(current_iter)

        self.optimizer_g.zero_grad()

        # print("\n==== OPTIMIZE PARAMETERS ====")
        # print(f"Input lq shape: {self.lq.shape}")
        # print(f"Using use_noise_map: {self.use_noise_map}")

        inputs = self._prepare_inputs(self.lq)

        if self.use_amp:
            with autocast():
                if self.use_texture_detection:
                    self.output, self.texture_mask = self.net_g(**inputs)
                else:
                    self.output, _ = self.net_g(**inputs)

                l_total = 0
                loss_dict = OrderedDict()

                # pixel loss
                if self.cri_pix:
                    if self.use_texture_detection and hasattr(self.cri_pix, 'forward') and 'texture_mask' in self.cri_pix.forward.__code__.co_varnames:
                        l_pix = self.cri_pix(self.output, self.gt, texture_mask=self.texture_mask)
                    else:
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


                l_total = l_total / self.accumulation_steps
                self.scaler.scale(l_total).backward()
                if (current_iter + 1) % self.accumulation_steps == 0:
                    self.scaler.step(self.optimizer_g)
                    self.scaler.update()

                self.log_dict = self.reduce_loss_dict(loss_dict)

                if self.ema_decay > 0:
                    self.model_ema(decay=self.ema_decay)
        else:

            if self.use_texture_detection:
                self.output, self.texture_mask = self.net_g(**inputs)

                if hasattr(self, 'texture_mask') and (current_iter % 1000 == 0 or current_iter < 4000 and current_iter % 500 == 0):

                    save_dir = os.path.join(self.opt['path']['experiments_root'], 'texture_masks')
                    self.visualize_texture_mask(self.texture_mask, current_iter, save_dir)
            else:
                self.output, _ = self.net_g(**inputs)

            # 感知损失计算前监控pred值
            if hasattr(self, 'cri_perceptual') and self.cri_perceptual is not None:
                pred_stats = {
                    'min': torch.min(self.output).item(),
                    'max': torch.max(self.output).item(),
                    'mean': torch.mean(self.output).item(),
                    'std': torch.std(self.output).item(),
                    'has_nan': torch.isnan(self.output).any().item(),
                    'has_inf': torch.isinf(self.output).any().item()
                }

                # 记录异常情况
                if pred_stats['max'] > 2.0 or pred_stats['has_nan'] or pred_stats['has_inf']:
                    print(f"\n[警告] 感知损失输入异常 (iter {current_iter}):")
                    print(f"输出统计: {pred_stats}")

            l_total = 0
            loss_dict = OrderedDict()

            # pixel loss
            if self.cri_pix:
                if self.use_texture_detection and hasattr(self.cri_pix, 'forward') and 'texture_mask' in self.cri_pix.forward.__code__.co_varnames:
                    l_pix = self.cri_pix(self.output, self.gt, texture_mask=self.texture_mask)
                else:
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

            # gradient explosion dectection
            self.detect_gradient_explosion(l_total, current_iter, loss_dict)

            l_total.backward()

            # TODO：需要调整max_norm
            logger = get_root_logger()

            self.adaptive_gradient_clipping(self.net_g.parameters(), current_iter, logger)

            self.optimizer_g.step()

            self.log_dict = self.reduce_loss_dict(loss_dict)

            if self.ema_decay > 0:
                self.model_ema(decay=self.ema_decay)

    def test(self):
        """Forward function used in testing."""
        # noise map
        inputs = self._prepare_inputs(self.lq)

        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():

                if self.use_texture_detection:
                    self.output, self.texture_mask = self.net_g_ema(**inputs)
                else:
                    self.output, _ = self.net_g_ema(**inputs)

                if self.corrector is not None:
                    self.output = self.corrector(self.output, self.gt)
        else:
            self.net_g.eval()
            with torch.no_grad():

                if self.use_texture_detection:
                    self.output, self.texture_mask = self.net_g(**inputs)
                else:
                    self.output, _ = self.net_g(**inputs)

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

    def generate_noise_map_from_metadata(self, image, noise_params):
        """Helper method to generate noise map from metadata."""
        from led.archs.dual_path_components.noise_map import generate_noise_map
        return generate_noise_map(
            image=image,
            noise_params=noise_params,
            camera_params=None,
            iso=None
        )

    def generate_noise_map_from_iso(self, image, iso):
        """Helper method to generate noise map from ISO value."""
        from led.archs.dual_path_components.noise_map import generate_noise_map
        return generate_noise_map(
            image=image,
            noise_params=None,
            camera_params=getattr(self, 'camera_params', None),
            iso=iso,
            camera_name=getattr(self, 'camera_name', None)
        )

    def _prepare_inputs(self, lq, noise_map=None):
        inputs = {'x': lq}

        # 处理噪声图
        if self.use_noise_map:
            # 情况1: 如果已提供噪声图，直接使用
            if noise_map is not None:
                inputs['noise_map'] = noise_map

            # 情况2: 如果有噪声生成器，使用它生成噪声图和处理图像
            # 这部分通常在训练阶段使用
            elif hasattr(self, 'noise_g') and self.training:
                self.camera_id = torch.randint(0, len(self.noise_g), (1,)).item()
                with torch.no_grad():
                    scale = self.white_level - self.black_level
                    self.gt = (self.gt - self.black_level) / scale
                    self.gt, self.lq, self.curr_metadata = self.noise_g(self.gt, scale, self.ratio, self.camera_id)

                    if 'cam' in self.curr_metadata:
                        self.camera_name = self.curr_metadata['cam']

                    # 生成噪声图
                    inputs['noise_map'] = self.generate_noise_map_from_metadata(self.lq, self.curr_metadata['noise_params'])

                    # 更新输入的lq
                    inputs['lq'] = self.lq

                    # 数据增强
                    if hasattr(self, 'augment') and self.augment is not None:
                        self.gt, inputs['lq'] = self.augment(self.gt, inputs['lq'])
                        if inputs['noise_map'] is not None:
                            # 对噪声图进行相同的增强
                            inputs['noise_map'] = self.augment(inputs['noise_map'])[0]

            # 情况3: 使用ISO生成噪声图（通常用于测试/推理）
            elif hasattr(self, 'iso') and self.iso is not None:
                inputs['noise_map'] = self.generate_noise_map_from_iso(lq, self.iso)

        return inputs

    # 在DualPathModel类中添加此方法
    def visualize_texture_mask(self, texture_mask, iteration, save_dir):
        """保存纹理掩码可视化结果，监控分布"""
        import matplotlib.pyplot as plt
        import os
        import numpy as np

        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        # 提取掩码数据并移至CPU
        mask_data = texture_mask.detach().cpu().numpy()
        values = mask_data.flatten()

        # 1. 保存掩码直方图
        plt.figure(figsize=(10, 6))
        hist, bins = np.histogram(values, bins=50)
        plt.bar(bins[:-1], hist, width=(bins[1]-bins[0])*0.8)
        plt.title(f"Texture Mask Distribution - Iter {iteration}")
        plt.xlabel("Mask Value")
        plt.ylabel("Frequency")
        plt.grid(True, alpha=0.3)

        # 添加分布统计信息到图表
        stats_text = (f"Mean: {values.mean():.4f}\n"
                    f"Std: {values.std():.4f}\n"
                    f"Min: {values.min():.4f}\n"
                    f"Max: {values.max():.4f}\n"
                    f"25%: {np.percentile(values, 25):.4f}\n"
                    f"50%: {np.percentile(values, 50):.4f}\n"
                    f"75%: {np.percentile(values, 75):.4f}")

        plt.figtext(0.15, 0.7, stats_text, bbox=dict(facecolor='white', alpha=0.8))

        plt.savefig(f"{save_dir}/mask_hist_{iteration}.png")
        plt.close()

        # 2. 保存掩码图像
        mask_vis = mask_data[0, 0]
        plt.figure(figsize=(8, 8))
        im = plt.imshow(mask_vis, cmap='viridis', vmin=0.1, vmax=0.7)
        plt.colorbar(im)
        plt.title(f"Texture Mask - Iter {iteration}")
        plt.savefig(f"{save_dir}/mask_vis_{iteration}.png")
        plt.close()

        # 3. 记录详细统计数据到文件
        with open(f"{save_dir}/mask_stats.txt", "a") as f:
            f.write(f"Iter {iteration}:\n")
            f.write(f"  Mean: {values.mean():.4f}, Std: {values.std():.4f}\n")
            f.write(f"  Min: {values.min():.4f}, Max: {values.max():.4f}\n")
            f.write(f"  Percentiles: 25%={np.percentile(values, 25):.4f}, "
                    f"50%={np.percentile(values, 50):.4f}, "
                    f"75%={np.percentile(values, 75):.4f}\n")

            # 掩码值分布表
            bins = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
            hist, _ = np.histogram(values, bins=bins)
            total = len(values)
            percentages = [count/total*100 for count in hist]

            f.write("  Value distribution:\n")
            for i, (count, percentage) in enumerate(zip(hist, percentages)):
                f.write(f"    {bins[i]:.1f}-{bins[i+1]:.1f}: {count} pixels ({percentage:.2f}%)\n")
            f.write("\n")

        # 4. 额外添加掩码分布情况到日志
        from led.utils import get_root_logger
        logger = get_root_logger()
        logger.info(f"Texture mask at iter {iteration}: "
                    f"mean={values.mean():.4f}, std={values.std():.4f}, "
                    f"min={values.min():.4f}, max={values.max():.4f}")

    def detect_gradient_explosion(self, loss, current_iter, loss_dict, threshold=100.0):

        is_critical = torch.isnan(loss) or torch.isinf(loss)
        is_high_loss = loss > threshold

        if is_critical or is_high_loss:
            logger = get_root_logger()

            if is_critical:
                error_msg = f"[!!!CRITICAL ERROR!!!] NaN/Inf loss detected: {loss.item()}, terminating training!"
                log_func = logger.error
            else:
                error_msg = f"[WARNING] High loss value detected: {loss.item()}, possible gradient instability!"
                log_func = logger.warning

            log_func(error_msg)
            log_func(f"Current iteration: {current_iter}, Learning rate: {self.get_current_learning_rate()}")

            # 检查纹理掩码
            if hasattr(self, 'texture_mask') and self.texture_mask is not None:
                mask_stats = {
                    "mean": torch.mean(self.texture_mask).item(),
                    "min": torch.min(self.texture_mask).item(),
                    "max": torch.max(self.texture_mask).item(),
                    "has_nan": torch.isnan(self.texture_mask).any().item()
                }
                log_func(f"Texture mask statistics: {mask_stats}")

                if mask_stats["mean"] > 0.6:
                    log_func(f"Texture mask mean is too high: {mask_stats['mean']:.4f}")

            # 检查各损失组件
            for loss_name, loss_value in loss_dict.items():
                log_func(f"Loss component {loss_name}: {loss_value.item()}")

            # 保存诊断信息
            debug_state = {
                'iter': current_iter,
                'loss': loss.item(),
                'loss_components': {k: v.item() for k, v in loss_dict.items()},
                'output_stats': {
                    'mean': torch.mean(self.output).item(),
                    'min': torch.min(self.output).item(),
                    'max': torch.max(self.output).item(),
                    'has_nan': torch.isnan(self.output).any().item()
                },
                'gt_stats': {
                    'mean': torch.mean(self.gt).item(),
                    'min': torch.min(self.gt).item(),
                    'max': torch.max(self.gt).item()
                }
            }

            # 保存调试状态到文件
            # debug_dir = os.path.join(self.opt['path']['experiments_root'], 'debug')
            # os.makedirs(debug_dir, exist_ok=True)

            # if is_critical:
            #     file_name = f'error_state_{current_iter}.pth'
            # else:
            #     file_name = f'warning_state_{current_iter}.pth'

            # torch.save(debug_state, os.path.join(debug_dir, file_name))

            # 只在NaN/Inf时终止训练
            if is_critical:
                raise RuntimeError(error_msg)

        return False

    def adaptive_gradient_clipping(self, parameters, current_iter, logger=None):
        """实施自适应分层梯度裁剪"""
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, float('inf'))

        # 定义分层裁剪策略
        if grad_norm < 10.0:
            # 正常范围内的梯度 - 不裁剪
            clip_norm = float('inf')
            action = "no clipping"
        elif grad_norm < 15.0:
            # 轻度超标 - 温和裁剪
            clip_norm = 10.0
            action = "mild clipping"
        else:
            # 重度超标 - 强力裁剪
            clip_norm = 8.0
            action = "strong clipping"

        # 应用裁剪
        if clip_norm != float('inf'):
            torch.nn.utils.clip_grad_norm_(parameters, max_norm=clip_norm)

        should_log = (current_iter % 100 == 0) or (grad_norm > 15.0)

        if should_log and logger is not None:
            if grad_norm > 15.0:
                logger.warning(f"Gradient norm at iter {current_iter}: {grad_norm:.4f}, applying {action} (max_norm={clip_norm})")
            else:
                logger.info(f"Gradient norm at iter {current_iter}: {grad_norm:.4f}, {action}")

        return grad_norm, clip_norm