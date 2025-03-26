import torch
import math

def generate_noise_map(image, noise_params=None, camera_params=None, iso=None, camera_name=None):
    """
    Generate standard deviation-based noise map

    Args:
        image: Input image (B, C, H, W)
        noise_params: Noise parameters containing K and sigma_r (used when available from noise generator)
        camera_params: Camera parameters (used for K computation from ISO)
        iso: ISO values of the input images (used when noise_params not available)
        camera_name: Camera model name to get specific parameters (e.g., 'SonyA7S2')

    Returns:
        noise_map: Noise map (B, C, H, W)
    """
    # Method 1: Use noise parameters directly from noise generator (preferred method)
    if noise_params is not None and 'shot' in noise_params:
        # Get noise parameters
        #TODOK应该是总增益，而不是单个增益？
        K = noise_params['shot']  # Corresponds to shot noise parameter K
        if 'read' in noise_params:
            sigma_r = noise_params['read'] if isinstance(noise_params['read'], torch.Tensor) else noise_params['read']['sigma']
        else:
            sigma_r = torch.tensor(0.01).to(image.device)

    # Method 2: Calculate K from ISO values (fallback method)
    elif iso is not None:
        if camera_params is not None and camera_name is not None and camera_name in camera_params:
            # Get K range from camera parameters
            K_min = camera_params[camera_name]['Kmin']
            K_max = camera_params[camera_name]['Kmax']

            # Get ISO range (typical values)
            ISO_min = 100
            ISO_max = 409600  # High value for Sony A7S2

            # Logarithmic mapping from ISO to K
            iso_tensor = iso.to(image.device)
            log_ratio = math.log(K_max/K_min) / math.log(ISO_max/ISO_min)
            K = K_min * torch.pow(iso_tensor / ISO_min, log_ratio)

            # Estimate sigma_r based on K
            sigma_r = 0.01 * torch.sqrt(K)
        else:
            # Simplified estimation without camera parameters
            iso_tensor = iso.to(image.device)
            K = iso_tensor / 100.0
            sigma_r = 0.01 * torch.sqrt(K)

        # Reshape K and sigma_r for proper broadcasting
        K = K.view(-1, 1, 1, 1)
        sigma_r = sigma_r.view(-1, 1, 1, 1)

    # If no valid parameters are available, return None
    else:
        return None

    # Calculate standard deviation noise map: sqrt(K*R + sigma_r^2)
    # Check and handle potential NaN values
    if torch.isnan(K).any():
        print("Warning: NaN values detected in K, replacing with default value 0.1")
        K = torch.nan_to_num(K, nan=0.1)
    if torch.isnan(sigma_r).any():
        print("Warning: NaN values detected in sigma_r, replacing with default value 0.01")
        sigma_r = torch.nan_to_num(sigma_r, nan=0.01)

    # For noise estimation, ensure values are non-negative
    image_for_noise = torch.abs(image)  # Use absolute values to preserve information

    # Calculate noise standard deviation: sqrt(K*R + sigma_r^2), ensuring inner term is positive
    inner_term = K * image_for_noise + sigma_r**2
    inner_term = torch.clamp(inner_term, min=1e-10)  # Ensure non-negative
    noise_std = torch.sqrt(inner_term)

    # Filter potential NaN values in the result
    if torch.isnan(noise_std).any():
        nan_percentage = torch.isnan(noise_std).float().mean().item() * 100
        print(f"Warning: {nan_percentage:.2f}% NaN values in noise_std, replaced")
        noise_std = torch.nan_to_num(noise_std, nan=0.01)

    return noise_std