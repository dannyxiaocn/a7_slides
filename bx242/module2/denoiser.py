import torch
import torch.nn as nn
import sys
import os
from typing import Union, Optional, Tuple

# Add parent directory to path to import UNet
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

class DenoiserWrapper:
    """Wrapper class for the pre-trained U-Net denoiser with per-channel standardization"""
    
    def __init__(self, model_path: str = 'denoiser.pth', device: Optional[str] = None, 
                 use_per_channel_norm: bool = True):
        """
        Initialize the denoiser wrapper
        
        Args:
            model_path: Path to the pre-trained model weights
            device: Device to run the model on ('cuda' or 'cpu')
            use_per_channel_norm: Whether to apply per-channel standardization
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        self.use_per_channel_norm = use_per_channel_norm
            
        # Import UNet from the root directory
        import importlib.util
        unet_path = os.path.join(os.path.dirname(__file__), '..', '..', 'UNet Model.py')
        spec = importlib.util.spec_from_file_location("unet_model", unet_path)
        if spec is None or spec.loader is None:
            raise ImportError("Could not load UNet model")
        unet_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(unet_module)
        Unet = unet_module.Unet
        
        # Initialize the model with the same parameters as specified in the instructions
        self.model = Unet(in_chans=3, out_chans=3, chans=64).to(self.device)
        
        # Load the pre-trained weights
        checkpoint_path = os.path.join(os.path.dirname(__file__), '..', '..', model_path)
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.model.eval()
        
    def compute_channel_stats(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute per-channel mean and standard deviation
        
        Args:
            image: Input image tensor of shape (B, C, H, W) or (C, H, W)
            
        Returns:
            Tuple of (channel_means, channel_stds)
        """
        if image.dim() == 3:
            # Shape: (C, H, W) -> compute stats over H, W dimensions
            mean = image.view(image.size(0), -1).mean(dim=1, keepdim=True)
            std = image.view(image.size(0), -1).std(dim=1, keepdim=True)
            # Reshape to (C, 1, 1) for broadcasting
            mean = mean.view(-1, 1, 1)
            std = std.view(-1, 1, 1)
        elif image.dim() == 4:
            # Shape: (B, C, H, W) -> compute stats over H, W dimensions for each batch
            mean = image.view(image.size(0), image.size(1), -1).mean(dim=2, keepdim=True)
            std = image.view(image.size(0), image.size(1), -1).std(dim=2, keepdim=True)
            # Reshape to (B, C, 1, 1) for broadcasting
            mean = mean.view(image.size(0), image.size(1), 1, 1)
            std = std.view(image.size(0), image.size(1), 1, 1)
        else:
            raise ValueError(f"Expected 3D or 4D tensor, got {image.dim()}D")
            
        # Add small epsilon to prevent division by zero
        std = torch.clamp(std, min=1e-6)
        
        return mean, std
        
    def normalize_channels(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply per-channel standardization: (image - mean) / std
        
        Args:
            image: Input image tensor
            
        Returns:
            Tuple of (normalized_image, channel_means, channel_stds)
        """
        if not self.use_per_channel_norm:
            return image, None, None
            
        mean, std = self.compute_channel_stats(image)
        normalized = (image - mean) / std
        
        return normalized, mean, std
        
    def denormalize_channels(self, normalized_image: torch.Tensor, 
                           mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        """
        Reverse per-channel standardization: normalized * std + mean
        
        Args:
            normalized_image: Normalized image tensor
            mean: Channel means used for normalization
            std: Channel standard deviations used for normalization
            
        Returns:
            Denormalized image tensor
        """
        if not self.use_per_channel_norm or mean is None or std is None:
            return normalized_image
            
        denormalized = normalized_image * std + mean
        return denormalized
        
    def denoise(self, image: torch.Tensor) -> torch.Tensor:
        """
        Apply denoising to an image with per-channel standardization
        
        Args:
            image: Input image tensor of shape (B, C, H, W) or (C, H, W)
            
        Returns:
            Denoised image tensor of the same shape
        """
        was_3d = False
        if image.dim() == 3:
            was_3d = True
            image = image.unsqueeze(0)  # Add batch dimension
            
        # Move to device
        image = image.to(self.device)
        
        # Apply per-channel standardization
        normalized_image, mean, std = self.normalize_channels(image)
            
        with torch.no_grad():
            denoised_normalized = self.model(normalized_image)
            
        # Denormalize the output
        denoised = self.denormalize_channels(denoised_normalized, mean, std)
            
        if was_3d:
            denoised = denoised.squeeze(0)  # Remove batch dimension
            
        return denoised
        
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """Make the wrapper callable"""
        return self.denoise(image)
        
    def set_normalization(self, use_per_channel_norm: bool):
        """
        Enable or disable per-channel normalization
        
        Args:
            use_per_channel_norm: Whether to use per-channel standardization
        """
        self.use_per_channel_norm = use_per_channel_norm
        
    def get_normalization_status(self) -> bool:
        """
        Get current normalization status
        
        Returns:
            Whether per-channel normalization is enabled
        """
        return self.use_per_channel_norm 