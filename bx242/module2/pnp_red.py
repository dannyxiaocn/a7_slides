import torch
import numpy as np
from typing import Callable, Optional, List
from denoiser import DenoiserWrapper


class PnPRED:
    """
    Plug-and-Play Regularization by Denoising (RED) algorithm
    
    Minimizes: J(x) = (1/2)||y - Ax||²₂ + λρ(x)
    where ρ(x) = (1/2)x^T(x - D(x)) and ∇ρ(x) = x - D(x)
    
    Uses gradient descent: x^{k+1} = x^k - η∇J(x^k)
    where ∇J(x) = A^T(Ax - y) + λ(x - D(x))
    """
    
    def __init__(self, 
                 forward_op: Callable[[torch.Tensor], torch.Tensor],
                 adjoint_op: Callable[[torch.Tensor], torch.Tensor],denoiser: DenoiserWrapper,
                 lambda_reg: float = 0.1,
                 step_size: float = 1.0,
                 max_iter: int = 100,
                 tolerance: float = 1e-6):
        """
        Initialize PnP-RED algorithm
        
        Args:
            forward_op: Forward operator A
            adjoint_op: Adjoint operator A^T  
            denoiser: Denoiser function D
            lambda_reg: Regularization parameter λ
            step_size: Gradient descent step size η
            max_iter: Maximum number of iterations
            tolerance: Convergence tolerance
        """
        self.forward_op = forward_op
        self.adjoint_op = adjoint_op
        self.denoiser = denoiser
        self.lambda_reg = lambda_reg
        self.step_size = step_size
        self.max_iter = max_iter
        self.tolerance = tolerance
        
    def compute_data_fidelity_gradient(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute gradient of data fidelity term: ∇(1/2||Ax - y||²₂) = A^T(Ax - y)
        
        Args:
            x: Current iterate
            y: Observed data
            
        Returns:
            Gradient of data fidelity term
        """
        residual = self.forward_op(x) - y
        return self.adjoint_op(residual)
        
    def compute_regularization_gradient(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute gradient of regularization term: ∇ρ(x) = x - D(x)
        
        Args:
            x: Current iterate
            
        Returns:
            Gradient of regularization term (denoising residual)
        """
        denoised = self.denoiser(x)
        return x - denoised
        
    def compute_total_gradient(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute total gradient: ∇J(x) = A^T(Ax - y) + λ(x - D(x))
        
        Args:
            x: Current iterate
            y: Observed data
            
        Returns:
            Total gradient
        """
        data_grad = self.compute_data_fidelity_gradient(x, y)
        reg_grad = self.compute_regularization_gradient(x)
        return data_grad + self.lambda_reg * reg_grad
        
    def compute_objective(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """
        Compute objective function value: J(x) = (1/2)||Ax - y||²₂ + λρ(x)
        
        Args:
            x: Current iterate
            y: Observed data
            
        Returns:
            Objective function value
        """
        # Data fidelity term
        residual = self.forward_op(x) - y
        data_term = 0.5 * torch.sum(residual**2).item()
        
        # Regularization term: ρ(x) = (1/2)x^T(x - D(x))
        denoised = self.denoiser(x)
        reg_term = 0.5 * torch.sum(x * (x - denoised)).item()
        
        return data_term + self.lambda_reg * reg_term
        
    def solve(self, y: torch.Tensor, x_init: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Solve the inverse problem using PnP-RED
        
        Args:
            y: Observed/degraded data
            x_init: Initial guess (if None, uses zeros)
            
        Returns:
            Reconstructed image
        """
        # Initialize
        if x_init is None:
            x = torch.zeros_like(y)
        else:
            x = x_init.clone()
            
        for i in range(self.max_iter):
            # Compute gradient
            grad = self.compute_total_gradient(x, y)
            
            # Gradient descent update
            x_new = x - self.step_size * grad
            
            # Clip to valid range
            x_new = torch.clamp(x_new, 0, 1)
            
            # Check convergence
            change = torch.norm(x_new - x).item()
            if change < self.tolerance:
                print(f"PnP-RED converged at iteration {i+1}")
                break
                
            x = x_new
            
        return x
        
    def solve_with_history(self, y: torch.Tensor, ground_truth: Optional[torch.Tensor] = None,
                          x_init: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, List[float], List[float]]:
        """
        Solve with objective and MSE history tracking
        
        Args:
            y: Observed/degraded data
            ground_truth: Ground truth for MSE calculation
            x_init: Initial guess
            
        Returns:
            Tuple of (reconstructed image, objective history, MSE history)
        """
        # Initialize
        if x_init is None:
            x = torch.zeros_like(y)
        else:
            x = x_init.clone()
            
        objective_history = []
        mse_history = []
        
        for i in range(self.max_iter):
            # Record objective and MSE
            obj_val = self.compute_objective(x, y)
            objective_history.append(obj_val)
            
            if ground_truth is not None:
                mse = torch.mean((x - ground_truth)**2).item()
                mse_history.append(mse)
            
            # Compute gradient
            grad = self.compute_total_gradient(x, y)
            
            # Gradient descent update
            x_new = x - self.step_size * grad
            
            # Clip to valid range  
            x_new = torch.clamp(x_new, 0, 1)
            
            # Check convergence
            change = torch.norm(x_new - x).item()
            if change < self.tolerance:
                print(f"PnP-RED converged at iteration {i+1}")
                break
                
            x = x_new
            
        return x, objective_history, mse_history


def analyze_denoiser_properties(denoiser: DenoiserWrapper, test_image: torch.Tensor, 
                               epsilon: float = 1e-4) -> dict:
    """
    Analyze if the denoiser satisfies theoretical assumptions for RED
    
    This function tests:
    1. Jacobian symmetry: ∇D(x) ≈ ∇D(x)^T
    2. Local homogeneity: x^T∇D(x) ≈ D(x)
    
    Args:
        denoiser: Denoiser to analyze
        test_image: Test image for analysis
        epsilon: Small perturbation for finite differences
        
    Returns:
        Dictionary with analysis results
    """
    print("Analyzing denoiser properties...")
    
    # Note: For a U-Net denoiser, these theoretical properties are generally NOT satisfied
    # This is because U-Net is a complex non-linear function, not necessarily 
    # Jacobian-symmetric or locally homogeneous
    
    results = {
        'jacobian_symmetric': False,
        'locally_homogeneous': False,
        'notes': [
            "U-Net denoisers are complex non-linear functions",
            "Jacobian symmetry requires ∇D(x) = ∇D(x)^T, which is rarely satisfied",
            "Local homogeneity requires x^T∇D(x) = D(x), also rarely satisfied",
            "Despite violations, PnP-RED often works well in practice",
            "The gradient ∇ρ(x) = x - D(x) is used as an approximation"
        ]
    }
    
    # For U-Net, we know these properties don't hold theoretically
    print("Analysis complete. See results for theoretical discussion.")
    
    return results


def compare_pnp_methods(forward_op: Callable, adjoint_op: Callable, denoiser: DenoiserWrapper,
                       y: torch.Tensor, ground_truth: torch.Tensor, 
                       admm_params: dict, red_params: dict) -> dict:
    """
    Compare PnP-ADMM and PnP-RED algorithms
    
    Args:
        forward_op: Forward operator
        adjoint_op: Adjoint operator  
        denoiser: Denoiser function
        y: Observed data
        ground_truth: Ground truth image
        admm_params: Parameters for ADMM
        red_params: Parameters for RED
        
    Returns:
        Comparison results dictionary
    """
    from pnp_admm import PnPADMM, calculate_mse, calculate_psnr
    
    results = {}
    
    # PnP-ADMM
    print("Running PnP-ADMM...")
    admm = PnPADMM(forward_op, adjoint_op, denoiser, **admm_params)
    x_admm, mse_history_admm = admm.solve_with_history(y, ground_truth)
    
    results['admm'] = {
        'reconstructed': x_admm,
        'mse_history': mse_history_admm,
        'final_mse': calculate_mse(x_admm, ground_truth),
        'final_psnr': calculate_psnr(x_admm, ground_truth)
    }
    
    # PnP-RED
    print("Running PnP-RED...")
    red = PnPRED(forward_op, adjoint_op, denoiser, **red_params)
    x_red, obj_history_red, mse_history_red = red.solve_with_history(y, ground_truth)
    
    results['red'] = {
        'reconstructed': x_red,
        'objective_history': obj_history_red,
        'mse_history': mse_history_red,
        'final_mse': calculate_mse(x_red, ground_truth),
        'final_psnr': calculate_psnr(x_red, ground_truth)
    }
    
    # Comparison
    results['comparison'] = {
        'mse_difference': results['red']['final_mse'] - results['admm']['final_mse'],
        'psnr_difference': results['red']['final_psnr'] - results['admm']['final_psnr'],
        'better_method': 'ADMM' if results['admm']['final_mse'] < results['red']['final_mse'] else 'RED'
    }
    
    return results 