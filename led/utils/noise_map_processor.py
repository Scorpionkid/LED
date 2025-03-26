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

def apply_to_module(tensor, module, noise_map=None, target_channels=None):
    """
    Process input tensor and noise map before passing to a module.

    This method handles NaN checking and channel adjustment for safe module execution.

    Args:
        tensor (Tensor): Input feature tensor
        module (nn.Module): Neural network module to apply
        noise_map (Tensor, optional): Noise map tensor
        target_channels (int, optional): Target channel count for noise map

    Returns:
        Tensor: Result from module execution
    """
    # Check for NaN in input tensor
    detect_nan(tensor, f"input to {module.__class__.__name__}")

    # Prepare noise map if provided
    if noise_map is not None:
        # Check for NaN in noise map
        detect_nan(noise_map, f"noise map for {module.__class__.__name__}")

        # Adjust channels if needed
        if target_channels is None:
            target_channels = tensor.size(1)

        if noise_map.size(1) != target_channels:
            noise_map = adjust_channels(noise_map, target_channels)

    # Apply module
    try:
        if noise_map is not None:
            result = module(tensor, noise_map)
        else:
            result = module(tensor)

        # Check for NaN in output
        detect_nan(result, f"output from {module.__class__.__name__}")
        return result

    except Exception as e:
        warnings.warn(f"Error applying module {module.__class__.__name__}: {str(e)}")
        # Return tensor to maintain data flow in case of error
        return tensor