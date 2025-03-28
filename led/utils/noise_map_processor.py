"""
Noise Map Processing Utility Module

Provides utility functions for handling noise maps in neural networks,
including channel adjustment, NaN detection, and multi-scale noise map creation.
"""

import inspect
import functools
import torch
import warnings

def adjust_channels(tensor, target_channels):
    """
    Adjust the number of channels in a tensor to match the target.

    Args:
        tensor (Tensor): Input tensor of shape [B, C, H, W]
        target_channels (int): Desired number of channels

    Returns:
        Tensor: Adjusted tensor with shape [B, target_channels, H, W]
    """
    if tensor is None:
        return None

    if tensor.size(1) == target_channels:
        return tensor

    # For single-channel tensors, expand to target channels
    if tensor.size(1) == 1:
        return tensor.expand(-1, target_channels, -1, -1)

    # For multi-channel tensors, average then expand
    avg_tensor = torch.mean(tensor, dim=1, keepdim=True)
    return avg_tensor.expand(-1, target_channels, -1, -1)

def detect_nan(tensor, name="tensor"):
    """
    Detect NaN values in a tensor and output warnings.

    Args:
        tensor (Tensor): Input tensor to check
        name (str): Name identifier for the tensor in warning messages

    Returns:
        bool: True if NaN values are detected, False otherwise
    """
    if tensor is None:
        return False

    nan_mask = torch.isnan(tensor)
    nan_count = torch.sum(nan_mask).item()

    if nan_count > 0:
        warnings.warn(f"Detected {nan_count} NaN values in {name} (shape: {tensor.shape})")
        # Print samples of NaN positions
        nan_indices = torch.nonzero(nan_mask, as_tuple=True)
        sample_indices = [indices[0].item() for indices in nan_indices]
        unique_samples = set(sample_indices)
        warnings.warn(f"NaN values found in {len(unique_samples)} different samples")
        return True

    return False

def create_multiscale_maps(noise_map, scales=[1, 2, 4, 8]):
    """
    Create multi-scale versions of a noise map for different network levels.

    Args:
        noise_map (Tensor): Input noise map tensor of shape [B, C, H, W]
        scales (list): List of downsampling scales

    Returns:
        dict: Dictionary containing noise maps at different scales
    """
    if noise_map is None:
        return {f'scale_{s}': None for s in scales}

    # Check for NaN values
    detect_nan(noise_map, "noise_map")

    result = {}

    for scale in scales:
        if scale == 1:
            result[f'scale_{scale}'] = noise_map
        else:
            # Apply average pooling for each scale
            downsampled = noise_map
            for _ in range(int(scale/2)):
                downsampled = torch.nn.functional.avg_pool2d(downsampled, 2)
            result[f'scale_{scale}'] = downsampled

    return result

def apply_to_module(tensor, module, noise_map=None, texture_mask=None,
                    target_channels=None):
    """
    Process input tensor and noise map before passing to a module.

    This method handles NaN checking and channel adjustment for safe module execution.

    Args:
        tensor (Tensor): Input feature tensor
        module (nn.Module): Neural network module to apply
        noise_map (Tensor, optional): Noise map tensor
        target_channels (int, optional): Target channel count for noise map
        texture_mask (Tensor, optional): Texture mask tensor

    Returns:
        Tensor: Result from module execution
    """
    module_name = module.__class__.__name__ if hasattr(module, '__class__') else type(module).__name__
    detect_nan(tensor, f"input to {module_name}")

    # Adjust channels if needed
    if target_channels is None:
        target_channels = tensor.size(1)

    # Prepare noise map if provided
    if noise_map is not None:
        # Check for NaN in noise map
        detect_nan(noise_map, f"noise map for {module.__class__.__name__}")

        # Check spatial dimensions - should match input tensor
        if noise_map.shape[2:] != tensor.shape[2:]:
            raise ValueError(f"Noise map spatial dimensions {noise_map.shape[2:]} do not match "
                        f"input tensor dimensions {tensor.shape[2:]} for module {module.__class__.__name__}")


    # Prepare texture_mask if provided
    if texture_mask is not None:
        # Check for NaN in texture mask
        detect_nan(texture_mask, f"texture mask for {module.__class__.__name__}")

        # Check spatial dimensions - should match input tensor
        if texture_mask.shape[2:] != tensor.shape[2:]:
            raise ValueError(f"Texture mask spatial dimensions {texture_mask.shape[2:]} do not match "
                        f"input tensor dimensions {tensor.shape[2:]} for module {module.__class__.__name__}")


    if hasattr(module, '__class__'):
        module_name = module.__class__.__name__

        if module_name in ['HighFrequencyAttention', 'AdaptiveDenoiseGate',
                          'DynamicFusion', 'SharpnessRecovery'] or module_name == 'function':
            if noise_map is not None and noise_map.size(1) > 1:
                noise_map = torch.mean(noise_map, dim=1, keepdim=True)

            if texture_mask is not None and texture_mask.size(1) > 1:
                texture_mask = torch.mean(texture_mask, dim=1, keepdim=True)
        elif target_channels is not None:
            if noise_map is not None and noise_map.size(1) != target_channels:
                noise_map = adjust_channels(noise_map, target_channels)

            if texture_mask is not None and texture_mask.size(1) != target_channels:
                texture_mask = adjust_channels(texture_mask, target_channels)


    # Apply module
    try:
        if not callable(module):
            raise ValueError(f"模块 {module} 不可调用")

        # 获取调用签名和参数
        if inspect.isfunction(module) or inspect.ismethod(module) or isinstance(module, functools.partial):
            # 函数、方法或partial对象
            sig = inspect.signature(module)
            param_names = list(sig.parameters.keys())
            param_count = len(param_names)

            # 检查是否是lambda/function - 这些通常使用位置参数
            is_lambda = module.__name__ == '<lambda>'
            is_local_func = 'local' in str(module)

            # lambda需要基于参数数量决定调用方式
            if (is_lambda or is_local_func):
                if param_count == 1:
                    return module(tensor)
                elif param_count == 2 and noise_map is not None:
                    return module(tensor, noise_map)
                elif param_count == 2 and texture_mask is not None:
                    return module(tensor, texture_mask)
                elif param_count == 3 and noise_map is not None and texture_mask is not None:
                    return module(tensor, noise_map, texture_mask)
                else:
                    # 如果无法确定如何调用，直接使用基本调用并让错误显示
                    return module(tensor)

        # 检查模块的调用参数
        if hasattr(module, 'forward'):
            # PyTorch模块 - 使用forward方法的参数
            sig_params = list(inspect.signature(module.forward).parameters.keys())
            if sig_params and sig_params[0] == 'self':
                sig_params = sig_params[1:]  # 移除self参数
        else:
            # 普通可调用对象
            sig_params = list(inspect.signature(module.__call__ if hasattr(module, '__call__') else module).parameters.keys())
            if sig_params and sig_params[0] == 'self':
                sig_params = sig_params[1:]  # 移除self参数

        # 根据参数决定调用方式
        if len(sig_params) >= 3 and 'noise_map' in sig_params and 'texture_mask' in sig_params:
            # 接受noise_map和texture_mask参数
            return module(tensor, noise_map, texture_mask)
        elif len(sig_params) >= 2 and 'noise_map' in sig_params:
            # 只接受noise_map参数
            return module(tensor, noise_map)
        elif len(sig_params) >= 2 and 'texture_mask' in sig_params:
            # 只接受texture_mask参数
            return module(tensor, texture_mask)
        else:
            # 基本模块，只接受一个输入
            return module(tensor)

    except Exception as e:
        # 详细的错误报告
        print(f"\n========== 执行中断用于调试 ==========")
        print(f"模块 {module_name} 中发生错误")
        print(f"输入张量形状: {tensor.shape}")
        print(f"噪声图: {'None' if noise_map is None else noise_map.shape}")
        print(f"纹理掩码: {'None' if texture_mask is None else texture_mask.shape}")
        print(f"目标通道: {target_channels}")
        print(f"错误: {str(e)}")

        # 提供额外信息用于调试
        if hasattr(module, 'forward'):
            try:
                forward_sig = inspect.signature(module.forward)
                print(f"模块forward方法签名: {forward_sig}")
            except:
                pass
        elif callable(module):
            try:
                call_sig = inspect.signature(module)
                print(f"可调用对象签名: {call_sig}")
            except:
                pass

        raise RuntimeError(f"调试中断: {module_name} 中的错误: {str(e)}")