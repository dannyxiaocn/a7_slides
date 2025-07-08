import torch
import numpy as np
from typing import Callable, Optional
from conjugate_gradient import conjugate_gradient
from denoiser import DenoiserWrapper


class PnPADMM:
    """Plug-and-Play ADMM algorithm for inverse problems"""
    
    def __init__(self, 
                 forward_op: Callable[[torch.Tensor], torch.Tensor],
                 adjoint_op: Callable[[torch.Tensor], torch.Tensor],
                 denoiser: DenoiserWrapper,
                 eta: float = 1e-4,
                 max_iter: int = 50,
                 cg_max_iter: int = 100,
                 cg_tol: float = 1e-6):
        """
        Initialize PnP-ADMM algorithm
        
        Args:
            forward_op: Forward operator A
            adjoint_op: Adjoint operator A^T
            denoiser: Denoiser function D
            eta: Regularization parameter
            max_iter: Maximum number of ADMM iterations
            cg_max_iter: Maximum iterations for conjugate gradient
            cg_tol: Tolerance for conjugate gradient
        """
        self.forward_op = forward_op
        self.adjoint_op = adjoint_op
        self.denoiser = denoiser
        self.eta = eta
        self.max_iter = max_iter
        self.cg_max_iter = cg_max_iter
        self.cg_tol = cg_tol
        
    def _create_system_operator(self, shape: torch.Size, device: torch.device) -> Callable[[torch.Tensor], torch.Tensor]:
        """
        Create the system operator C = A^T A + eta * I for the x-update step
        
        Args:
            shape: Shape of the input tensor
            device: Device to run operations on
            
        Returns:
            System operator function
        """
        def system_op(x: torch.Tensor) -> torch.Tensor:
            return self.adjoint_op(self.forward_op(x)) + self.eta * x
        
        return system_op
        
    def solve(self, y: torch.Tensor, x_init: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Solve the inverse problem using PnP-ADMM
        
        Args:
            y: Observed/degraded image
            x_init: Initial guess (if None, uses zeros)
            
        Returns:
            Reconstructed image
        """
        device = y.device
        
        # Initialize variables
        if x_init is None:
            x = torch.zeros_like(y)
        else:
            x = x_init.clone()
            
        v = torch.zeros_like(y)
        u = torch.zeros_like(y)
        
        # Create system operator for CG
        system_op = self._create_system_operator(y.shape, device)
        
        mse_history = []
        
        for i in range(self.max_iter):
            # x-update: solve (A^T A + eta I) x = A^T y + eta (v - u)
            rhs = self.adjoint_op(y) + self.eta * (v - u)
            x = conjugate_gradient(system_op, rhs, x.clone(), self.cg_max_iter, self.cg_tol)
            
            # v-update: v = D(x + u)
            v = self.denoiser(x + u)
            
            # u-update: u = u + (x - v)
            u = u + (x - v)
            
            # Clip to valid range
            x = torch.clamp(x, 0, 1)
            
        return x
        
    def solve_with_history(self, y: torch.Tensor, ground_truth: Optional[torch.Tensor] = None, 
                          x_init: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, list]:
        """
        Solve with MSE history tracking
        
        Args:
            y: Observed/degraded image
            ground_truth: Ground truth image for MSE calculation
            x_init: Initial guess
            
        Returns:
            Tuple of (reconstructed image, MSE history)
        """
        device = y.device
        
        # Initialize variables
        if x_init is None:
            x = torch.zeros_like(y)
        else:
            x = x_init.clone()
            
        v = torch.zeros_like(y)
        u = torch.zeros_like(y)
        
        # Create system operator for CG
        system_op = self._create_system_operator(y.shape, device)
        
        mse_history = []
        
        for i in range(self.max_iter):
            # x-update: solve (A^T A + eta I) x = A^T y + eta (v - u)
            rhs = self.adjoint_op(y) + self.eta * (v - u)
            x = conjugate_gradient(system_op, rhs, x.clone(), self.cg_max_iter, self.cg_tol)
            
            # v-update: v = D(x + u)
            v = self.denoiser(x + u)
            
            # u-update: u = u + (x - v)
            u = u + (x - v)
            
            # Clip to valid range
            x = torch.clamp(x, 0, 1)
            
            # Calculate MSE if ground truth is provided
            if ground_truth is not None:
                mse = torch.mean((x - ground_truth) ** 2).item()
                mse_history.append(mse)
            
        return x, mse_history


def calculate_mse(image1: torch.Tensor, image2: torch.Tensor) -> float:
    """Calculate Mean Squared Error between two images"""
    return torch.mean((image1 - image2) ** 2).item()


def calculate_psnr(image1: torch.Tensor, image2: torch.Tensor, max_val: float = 1.0) -> float:
    """Calculate Peak Signal-to-Noise Ratio between two images"""
    mse = calculate_mse(image1, image2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(max_val / np.sqrt(mse)) 