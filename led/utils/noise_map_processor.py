"""
Noise Map Processing Utility Module

Provides utility functions for handling noise maps in neural networks,
including channel adjustment, NaN detection, and multi-scale noise map creation.
"""

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
    module_name = module.__class__.__name__
    # print(f"\n--- APPLY_TO_MODULE: {module_name} ---")
    # print(f"Input tensor shape: {tensor.shape}")
    # print(f"Noise map: {'None' if noise_map is None else noise_map.shape}")
    # print(f"Target channels: {target_channels}")

    # Check for NaN in input tensor
    detect_nan(tensor, f"input to {module.__class__.__name__}")

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


    # if hasattr(module, 'forward'):
    #     params = module.forward.__code__.co_varnames

    #     # 检查模块的类名，为不同类型模块采用不同处理策略
    #     module_name = module.__class__.__name__

    #     # 对特定模块保持掩码的单通道特性
    #     if module_name in ['HighFrequencyAttention', 'AdaptiveDenoiseGate',
    #                        'DynamicFusion', 'SharpnessRecovery', 'function']:
            # 这些模块需要单通道掩码，不进行通道扩展
        if noise_map is not None and noise_map.size(1) > 1:
            # 如果已经扩展了通道，取第一个通道或平均
            noise_map = torch.mean(noise_map, dim=1, keepdim=True)

        if texture_mask is not None and texture_mask.size(1) > 1:
            texture_mask = torch.mean(texture_mask, dim=1, keepdim=True)
        # elif target_channels is not None:
            # 其他模块可能需要通道扩展
            # if noise_map is not None and noise_map.size(1) != target_channels:
            #     noise_map = adjust_channels(noise_map, target_channels)

            # if texture_mask is not None and texture_mask.size(1) != target_channels:
            #     texture_mask = adjust_channels(texture_mask, target_channels)


    # Apply module
    try:
        # Determine which forward signature to use based on module interface
        # if hasattr(module, 'forward'):
        #     params = module.forward.__code__.co_varnames

            # if 'texture_mask' in params and 'noise_map' in params:
                # Module accepts both texture_mask and noise_map
        result = module(tensor, noise_map, texture_mask)
            # elif 'texture_mask' in params:
            #     # Module accepts texture_mask but not noise_map
            #     result = module(tensor, texture_mask)
            # elif 'noise_map' in params:
            #     # Module accepts noise_map but not texture_mask
            #     result = module(tensor, noise_map)
            # else:
            #     # Basic module with no special parameters
            #     result = module(tensor)
        # else:
        #     # Fallback for callable objects without explicit forward method
        #     result = module(tensor)

        # Check for NaN in output
        detect_nan(result, f"output from {module.__class__.__name__}")
        return result

    except Exception as e:
        print("\n========== EXECUTION HALTED FOR DEBUGGING ==========")
        print(f"Error occurred in module {module_name}")
        print(f"Input tensor shape: {tensor.shape}")
        print(f"Noise map: {'None' if noise_map is None else noise_map.shape}")
        print("Stack trace:")
        # import traceback
        # traceback.print_exc()

        raise RuntimeError(f"Debug halt: Error in {module_name}: {str(e)}")