import torch
import torch.nn as nn
from typing import Tuple, Optional


def conv2d_block(kernel: torch.Tensor, channels: int, p: int, device: torch.device, stride: int = 1) -> Tuple[nn.Module, nn.Module]:
    """
    Returns nn.Conv2d and nn.ConvTranspose2d modules from 2D kernel, such that
    nn.ConvTranspose2d is the adjoint operator of nn.Conv2d
    
    Args:
        kernel: 2D kernel, p x p is the kernel size
        channels: number of image channels
        p: kernel size
        device: device to put the operators on
        stride: stride for convolution
        
    Returns:
        Tuple of (forward operator, adjoint operator)
    """
    kernel_size = (kernel.shape[0], kernel.shape[1])  # Convert to tuple
    kernel = kernel / kernel.sum()  # Normalize kernel
    kernel = kernel.repeat(channels, 1, 1, 1)  # Repeat for each channel
    
    # Forward operator (convolution)
    filter_forward = nn.Conv2d(
        in_channels=channels, out_channels=channels,
        kernel_size=kernel_size, groups=channels, bias=False, stride=stride,
        padding=p//2
    )
    filter_forward.weight.data = kernel
    filter_forward.weight.requires_grad = False
    
    # Adjoint operator (transpose convolution)
    filter_adjoint = nn.ConvTranspose2d(
        in_channels=channels, out_channels=channels,
        kernel_size=kernel_size, groups=channels, bias=False, stride=stride,
        padding=p//2,
    )
    filter_adjoint.weight.data = kernel
    filter_adjoint.weight.requires_grad = False
    
    return filter_forward.to(device), filter_adjoint.to(device)


def create_motion_blur_kernel(p: int, device: torch.device) -> torch.Tensor:
    """
    Create a motion blur kernel of size p x p with all entries equal to 1/p^2
    
    Args:
        p: kernel size
        device: device to put the kernel on
        
    Returns:
        Motion blur kernel tensor
    """
    kernel = torch.ones(p, p, device=device) / (p * p)
    return kernel


class MotionBlurOperator:
    """Class to handle motion blur forward and adjoint operations"""
    
    def __init__(self, p: int, channels: int = 3, device: Optional[torch.device] = None):
        """
        Initialize motion blur operator
        
        Args:
            p: kernel size for motion blur
            channels: number of image channels
            device: device to run operations on
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.p = p
        self.channels = channels
        self.device = device
        
        # Create motion blur kernel
        kernel = create_motion_blur_kernel(p, device)
        
        # Create forward and adjoint operators
        self.forward_op, self.adjoint_op = conv2d_block(kernel, channels, p, device)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply forward blur operation"""
        return self.forward_op(x)
        
    def adjoint(self, x: torch.Tensor) -> torch.Tensor:
        """Apply adjoint blur operation"""
        return self.adjoint_op(x)
        
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Make operator callable (applies forward operation)"""
        return self.forward(x) 