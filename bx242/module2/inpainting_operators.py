import torch
from typing import Optional, Tuple


class InpaintingOperator:
    """
    Inpainting operator that applies a binary mask to simulate missing pixels.
    Forward operator: y = M ⊙ x (element-wise multiplication)
    Adjoint operator: A^T = A (same as forward for inpainting)
    """
    
    def __init__(self, mask: torch.Tensor, device: Optional[torch.device] = None):
        """
        Initialize inpainting operator with a binary mask
        
        Args:
            mask: Binary mask tensor where 0 = missing pixel, 1 = observed pixel
            device: Device to run operations on
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
        self.device = device
        self.mask = mask.to(device)
        self.missing_ratio = 1.0 - (mask.sum().float() / mask.numel()).item()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply forward inpainting operation (mask application)
        
        Args:
            x: Input image tensor
            
        Returns:
            Masked image tensor
        """
        return x * self.mask
        
    def adjoint(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply adjoint inpainting operation (same as forward for inpainting)
        
        Args:
            x: Input image tensor
            
        Returns:
            Masked image tensor
        """
        return x * self.mask
        
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Make operator callable (applies forward operation)"""
        return self.forward(x)
        
    def get_missing_ratio(self) -> float:
        """Get the ratio of missing pixels"""
        return self.missing_ratio


def create_random_mask(shape: Tuple[int, ...], missing_ratio: float, 
                      device: Optional[torch.device] = None, seed: Optional[int] = None) -> torch.Tensor:
    """
    Create a random binary mask for inpainting
    
    Args:
        shape: Shape of the mask tensor (e.g., (1, 3, H, W))
        missing_ratio: Ratio of pixels to be missing (0.0 to 1.0)
        device: Device to create mask on
        seed: Random seed for reproducible results
        
    Returns:
        Binary mask tensor where 0 = missing pixel, 1 = observed pixel
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    if seed is not None:
        torch.manual_seed(seed)
        
    # Generate random mask
    random_values = torch.rand(shape, device=device)
    
    # Create binary mask: 1 where we keep pixels, 0 where we remove them
    mask = (random_values > missing_ratio).float()
    
    return mask


def create_structured_mask(shape: Tuple[int, ...], pattern: str = 'checkerboard',
                          missing_ratio: float = 0.5, device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Create structured masks for inpainting (optional, for testing different patterns)
    
    Args:
        shape: Shape of the mask tensor (e.g., (1, 3, H, W))
        pattern: Type of structured pattern ('checkerboard', 'stripes_h', 'stripes_v', 'center_square')
        missing_ratio: Approximate ratio of missing pixels
        device: Device to create mask on
        
    Returns:
        Binary mask tensor
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    batch_size, channels, height, width = shape
    mask = torch.ones(shape, device=device)
    
    if pattern == 'checkerboard':
        # Create checkerboard pattern
        for i in range(height):
            for j in range(width):
                if (i + j) % 2 == 0:
                    mask[:, :, i, j] = 0
                    
    elif pattern == 'stripes_h':
        # Horizontal stripes
        stripe_width = max(1, int(2 / missing_ratio))
        for i in range(0, height, stripe_width):
            mask[:, :, i:i+int(stripe_width*missing_ratio), :] = 0
            
    elif pattern == 'stripes_v':
        # Vertical stripes
        stripe_width = max(1, int(2 / missing_ratio))
        for j in range(0, width, stripe_width):
            mask[:, :, :, j:j+int(stripe_width*missing_ratio)] = 0
            
    elif pattern == 'center_square':
        # Remove center square
        center_h, center_w = height // 2, width // 2
        size = int((height * width * missing_ratio) ** 0.5)
        h_start = max(0, center_h - size // 2)
        h_end = min(height, center_h + size // 2)
        w_start = max(0, center_w - size // 2)
        w_end = min(width, center_w + size // 2)
        mask[:, :, h_start:h_end, w_start:w_end] = 0
        
    return mask


def visualize_mask(mask: torch.Tensor, title: str = "Inpainting Mask") -> None:
    """
    Visualize an inpainting mask
    
    Args:
        mask: Binary mask tensor
        title: Title for the plot
    """
    import matplotlib.pyplot as plt
    
    # Convert to numpy for visualization
    if mask.dim() == 4:  # (B, C, H, W)
        mask_vis = mask[0, 0].cpu().numpy()  # Take first batch and channel
    elif mask.dim() == 3:  # (C, H, W) 
        mask_vis = mask[0].cpu().numpy()  # Take first channel
    else:  # (H, W)
        mask_vis = mask.cpu().numpy()
        
    missing_ratio = 1.0 - mask_vis.mean()
    
    plt.figure(figsize=(8, 6))
    plt.imshow(mask_vis, cmap='gray', vmin=0, vmax=1)
    plt.title(f"{title}\nMissing Ratio: {missing_ratio:.1%} (White=Keep, Black=Missing)")
    plt.colorbar()
    plt.axis('off')
    plt.show()


# Utility functions for mask analysis
def analyze_mask(mask: torch.Tensor) -> dict:
    """
    Analyze properties of an inpainting mask
    
    Args:
        mask: Binary mask tensor
        
    Returns:
        Dictionary with mask statistics
    """
    total_pixels = mask.numel()
    observed_pixels = mask.sum().item()
    missing_pixels = total_pixels - observed_pixels
    missing_ratio = missing_pixels / total_pixels
    
    return {
        'total_pixels': total_pixels,
        'observed_pixels': int(observed_pixels),
        'missing_pixels': int(missing_pixels),
        'missing_ratio': missing_ratio,
        'observed_ratio': 1.0 - missing_ratio
    } 