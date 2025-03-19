import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveDenoiseGate(nn.Module):
    """Adaptive gating mechanism to adjust denoising strength based on local noise characteristics"""
    def __init__(self, channels):
        super(AdaptiveDenoiseGate, self).__init__()
        self.noise_est = nn.Conv2d(channels, 1, 3, padding=1)
        self.gate = nn.Sequential(
            nn.Conv2d(channels, channels//2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels//2, channels, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x, features=None):
        """
        Args:
            x: input features
            features: optional additional features for estimating noise level
        """
        if features is None:
            features = x

        # Estimate noise level
        noise_level = self.noise_est(features)

        # Generate denoising strength gate
        denoise_strength = self.gate(noise_level)

        return denoise_strength

class ResidualDenoiser(nn.Module):
    """Residual learning structure to learn noise residuals"""
    def __init__(self, channels):
        super(ResidualDenoiser, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x, gate=None):
        """
        Args:
            x: input features
            gate: optional gating signal
        """
        residual = self.activation(self.conv1(x))
        residual = self.conv2(residual)

        if gate is not None:
            # If gating signal is provided, adjust residual based on gating signal
            return x + residual * gate
        else:
            return