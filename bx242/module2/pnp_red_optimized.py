"""
Optimized PnP-RED Implementation

This module provides improved versions of PnP-RED with better parameters,
stability enhancements, and adaptive strategies for better visual quality.
"""

import torch
import numpy as np
from typing import Callable, Optional, List, Tuple
from denoiser import DenoiserWrapper
from pnp_red import PnPRED


class PnPREDOptimized(PnPRED):
    """
    Optimized PnP-RED with improved stability and adaptive parameters
    """
    
    def __init__(self, 
                 forward_op: Callable[[torch.Tensor], torch.Tensor],
                 adjoint_op: Callable[[torch.Tensor], torch.Tensor],
                 denoiser: DenoiserWrapper,
                 lambda_reg: float = 0.05,  # Reduced from 0.1 for better stability
                 step_size: float = 0.1,    # Reduced from 1.0 for stability
                 max_iter: int = 100,
                 tolerance: float = 1e-6,
                 adaptive_step: bool = True,
                 line_search: bool = True):
        """
        Initialize optimized PnP-RED
        
        Args:
            adaptive_step: Whether to use adaptive step size
            line_search: Whether to use line search for step size
        """
        super().__init__(forward_op, adjoint_op, denoiser, lambda_reg, 
                        step_size, max_iter, tolerance)
        self.adaptive_step = adaptive_step
        self.line_search = line_search
        self.initial_step_size = step_size
        
    def line_search_step_size(self, x: torch.Tensor, y: torch.Tensor, 
                             grad: torch.Tensor, initial_step: Optional[float] = None) -> float:
        """
        Simple backtracking line search for optimal step size
        """
        if initial_step is None:
            initial_step = self.step_size
            
        current_obj = self.compute_objective(x, y)
        step = initial_step
        
        for _ in range(10):  # Max 10 line search iterations
            x_new = torch.clamp(x - step * grad, 0, 1)
            new_obj = self.compute_objective(x_new, y)
            
            if new_obj < current_obj:
                return step
            step *= 0.5
            
        return step
        
    def solve_optimized(self, y: torch.Tensor, x_init: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Solve with optimizations
        """
        # Better initialization: start from masked observation
        if x_init is None:
            x = y.clone()  # Start from observed data instead of zeros
        else:
            x = x_init.clone()
            
        current_step_size = self.initial_step_size
        
        for i in range(self.max_iter):
            # Compute gradient
            grad = self.compute_total_gradient(x, y)
            
            # Adaptive step size or line search
            if self.line_search:
                step = self.line_search_step_size(x, y, grad, current_step_size)
            elif self.adaptive_step:
                # Simple adaptive rule: decrease if oscillating
                if i > 0 and i % 10 == 0:
                    current_step_size *= 0.9
                step = current_step_size
            else:
                step = self.step_size
                
            # Gradient descent update
            x_new = x - step * grad
            
            # Clip to valid range
            x_new = torch.clamp(x_new, 0, 1)
            
            # Check convergence
            change = torch.norm(x_new - x).item()
            if change < self.tolerance:
                print(f"Optimized PnP-RED converged at iteration {i+1}")
                break
                
            x = x_new
            
        return x


def compare_pnp_red_variants(forward_op: Callable, adjoint_op: Callable, 
                           denoiser: DenoiserWrapper, y: torch.Tensor, 
                           ground_truth: torch.Tensor) -> dict:
    """
    Compare different PnP-RED parameter settings
    """
    from pnp_admm import calculate_mse, calculate_psnr
    
    print("Testing different PnP-RED parameter combinations...")
    
    # Parameter combinations to test
    param_sets = [
        {'lambda_reg': 0.1, 'step_size': 1.0, 'name': 'Original (Instructions)'},
        {'lambda_reg': 0.05, 'step_size': 0.1, 'name': 'Conservative'},
        {'lambda_reg': 0.01, 'step_size': 0.05, 'name': 'Very Conservative'},
        {'lambda_reg': 0.05, 'step_size': 0.1, 'adaptive_step': True, 'name': 'Adaptive'},
        {'lambda_reg': 0.05, 'step_size': 0.1, 'line_search': True, 'name': 'Line Search'},
    ]
    
    results = {}
    
    for params in param_sets:
        name = params.pop('name')
        print(f"\nTesting {name}: {params}")
        
        try:
            if 'adaptive_step' in params or 'line_search' in params:
                # Use optimized version
                red = PnPREDOptimized(forward_op, adjoint_op, denoiser, 
                                    max_iter=30, **params)
                x_reconstructed = red.solve_optimized(y)
            else:
                # Use standard version
                red = PnPRED(forward_op, adjoint_op, denoiser, 
                           max_iter=30, **params)
                x_reconstructed = red.solve(y)
            
            mse = calculate_mse(x_reconstructed, ground_truth)
            psnr = calculate_psnr(x_reconstructed, ground_truth)
            
            results[name] = {
                'reconstructed': x_reconstructed,
                'mse': mse,
                'psnr': psnr,
                'params': params,
                'success': True
            }
            
            print(f"  MSE: {mse:.6f}, PSNR: {psnr:.2f} dB")
            
        except Exception as e:
            print(f"  Failed: {e}")
            results[name] = {'success': False, 'error': str(e)}
    
    return results


def recommend_best_parameters(comparison_results: dict) -> dict:
    """
    Recommend best parameters based on comparison results
    """
    successful_results = {k: v for k, v in comparison_results.items() if v.get('success', False)}
    
    if not successful_results:
        return {'recommendation': 'All variants failed'}
    
    # Find best MSE
    best_mse = min(successful_results.values(), key=lambda x: x['mse'])
    best_mse_name = [k for k, v in successful_results.items() if v['mse'] == best_mse['mse']][0]
    
    return {
        'best_variant': best_mse_name,
        'best_mse': best_mse['mse'],
        'best_psnr': best_mse['psnr'],
        'recommended_params': best_mse['params'],
        'all_results': successful_results
    }


def visualize_parameter_comparison(comparison_results: dict, ground_truth: torch.Tensor, 
                                 y_corrupted: torch.Tensor) -> None:
    """
    Visualize comparison of different parameter settings
    """
    import matplotlib.pyplot as plt
    
    successful_results = {k: v for k, v in comparison_results.items() if v.get('success', False)}
    n_results = len(successful_results)
    
    if n_results == 0:
        print("No successful results to visualize")
        return
    
    fig, axes = plt.subplots(2, max(3, n_results), figsize=(4*max(3, n_results), 8))
    
    # Ground truth and corrupted (first row)
    axes[0, 0].imshow(ground_truth.squeeze().cpu().permute(1, 2, 0))
    axes[0, 0].set_title('Ground Truth')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(y_corrupted.squeeze().cpu().permute(1, 2, 0))
    axes[0, 1].set_title('Corrupted')
    axes[0, 1].axis('off')
    
    # Hide unused axes in first row
    for j in range(2, max(3, n_results)):
        axes[0, j].axis('off')
    
    # Results comparison (second row)
    for i, (name, result) in enumerate(successful_results.items()):
        axes[1, i].imshow(result['reconstructed'].squeeze().cpu().permute(1, 2, 0))
        axes[1, i].set_title(f'{name}\nMSE: {result["mse"]:.6f}\nPSNR: {result["psnr"]:.2f} dB')
        axes[1, i].axis('off')
    
    # Hide unused axes in second row
    for j in range(len(successful_results), max(3, n_results)):
        axes[1, j].axis('off')
    
    plt.tight_layout()
    plt.show()


# Utility function for the notebook
def run_optimization_study(inpaint_op, y_corrupted, ground_truth, denoiser):
    """
    Run complete optimization study for PnP-RED
    """
    print("="*60)
    print("PnP-RED OPTIMIZATION STUDY")
    print("="*60)
    
    # Compare different parameter settings
    comparison = compare_pnp_red_variants(
        inpaint_op.forward, inpaint_op.adjoint, denoiser, 
        y_corrupted, ground_truth
    )
    
    # Get recommendations
    recommendations = recommend_best_parameters(comparison)
    
    print("\n" + "="*60)
    print("OPTIMIZATION RECOMMENDATIONS:")
    print("="*60)
    
    if 'best_variant' in recommendations:
        print(f"Best performing variant: {recommendations['best_variant']}")
        print(f"Best MSE: {recommendations['best_mse']:.6f}")
        print(f"Best PSNR: {recommendations['best_psnr']:.2f} dB")
        print(f"Recommended parameters: {recommendations['recommended_params']}")
    
    # Visualize results
    visualize_parameter_comparison(comparison, ground_truth, y_corrupted)
    
    return comparison, recommendations 