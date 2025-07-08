"""
PnP-ADMM with Early Stopping Implementation

This module implements PnP-ADMM with early stopping mechanisms to prevent overfitting
as identified in Exercise 2.3.
"""

import torch
import torch.nn.functional as F
import numpy as np
from pnp_admm import PnPADMM
from conjugate_gradient import conjugate_gradient

def mse_loss(x, y):
    """Compute Mean Squared Error between two tensors."""
    return F.mse_loss(x, y).item()

class PnPADMMWithEarlyStopping(PnPADMM):
    """
    PnP-ADMM with Early Stopping to prevent overfitting.
    
    This class extends the basic PnP-ADMM implementation with various stopping criteria
    to avoid overfitting and degradation of reconstruction quality.
    """
    
    def __init__(self, forward_op, adjoint_op, denoiser, eta=1e-4, 
                 max_iter=50, cg_max_iter=10, cg_tol=1e-6,
                 patience=10, min_delta=1e-6, eta_decay=0.0, 
                 convergence_tol=1e-6):
        """
        Initialize PnP-ADMM with early stopping.
        
        Args:
            patience (int): Number of iterations to wait for improvement before stopping
            min_delta (float): Minimum change in MSE to consider as improvement
            eta_decay (float): Decay factor for eta (0 means no decay)
            convergence_tol (float): Tolerance for convergence based on solution change
        """
        super().__init__(forward_op, adjoint_op, denoiser, eta, 
                        max_iter, cg_max_iter, cg_tol)
        
        self.patience = patience
        self.min_delta = min_delta
        self.eta_decay = eta_decay
        self.eta_initial = eta
        self.convergence_tol = convergence_tol
        
    def solve_with_early_stopping(self, y_degraded, ground_truth=None, 
                                 verbose=True, stopping_mode='mse'):
        """
        Solve with early stopping based on different criteria.
        
        Args:
            y_degraded: Degraded observations
            ground_truth: Ground truth for MSE-based stopping (optional)
            verbose: Whether to print progress
            stopping_mode: 'mse', 'convergence', or 'combined'
            
        Returns:
            Tuple of (best_reconstruction, actual_iterations, stopping_reason)
        """
        device = y_degraded.device
        
        # Initialize variables (following base class pattern)
        x = torch.zeros_like(y_degraded)
        v = torch.zeros_like(y_degraded)
        u = torch.zeros_like(y_degraded)
        
        # Create system operator for CG
        system_op = self._create_system_operator(y_degraded.shape, device)
        
        # Early stopping variables
        best_mse = float('inf')
        best_x = x.clone()
        patience_counter = 0
        mse_history = []
        convergence_history = []
        
        # Iteration loop
        for k in range(self.max_iter):
            x_prev = x.clone()
            
            # Update eta with decay
            if self.eta_decay > 0:
                current_eta = self.eta_initial / (1 + self.eta_decay * k)
                self.eta = current_eta
                # Recreate system operator with new eta
                system_op = self._create_system_operator(y_degraded.shape, device)
            
            # Standard PnP-ADMM updates (following base class pattern)
            # x-update: solve (A^T A + η I) x = A^T y + η (v - u)
            rhs = self.adjoint_op(y_degraded) + self.eta * (v - u)
            x = conjugate_gradient(system_op, rhs, x.clone(), self.cg_max_iter, self.cg_tol)
            
            # v-update: v = D(x + u)
            v = self.denoiser(x + u)
            
            # u-update: u = u + (x - v)
            u = u + (x - v)
            
            # Clip to valid range
            x = torch.clamp(x, 0, 1)
            
            # Compute metrics for early stopping
            if ground_truth is not None and stopping_mode in ['mse', 'combined']:
                current_mse = mse_loss(x, ground_truth)
                mse_history.append(current_mse)
                
                # Check MSE-based early stopping
                if current_mse < best_mse - self.min_delta:
                    best_mse = current_mse
                    best_x = x.clone()
                    patience_counter = 0
                else:
                    patience_counter += 1
            
            # Convergence-based stopping
            if stopping_mode in ['convergence', 'combined']:
                conv_metric = torch.norm(x - x_prev) / (torch.norm(x) + 1e-8)
                convergence_history.append(conv_metric.item())
                
                if conv_metric < self.convergence_tol:
                    if verbose:
                        print(f"Converged at iteration {k+1}: relative change = {conv_metric:.2e}")
                    return x, k+1, 'convergence', {
                        'mse_history': mse_history,
                        'convergence_history': convergence_history
                    }
            
            # Early stopping check
            if ground_truth is not None and patience_counter >= self.patience:
                if verbose:
                    print(f"Early stopping at iteration {k+1}: no improvement for {self.patience} iterations")
                return best_x, k+1, 'early_stopping', {
                    'mse_history': mse_history,
                    'convergence_history': convergence_history,
                    'best_mse': best_mse
                }
            
            if verbose and (k + 1) % 10 == 0:
                status = f"Iter {k+1}/{self.max_iter}"
                if ground_truth is not None:
                    status += f", MSE: {current_mse:.6f}"
                if self.eta_decay > 0:
                    status += f", η: {self.eta:.2e}"
                print(status)
        
        # Return final result
        final_x = best_x if ground_truth is not None else x
        return final_x, self.max_iter, 'max_iterations', {
            'mse_history': mse_history,
            'convergence_history': convergence_history
        }

class AdaptiveEtaPnPADMM(PnPADMM):
    """
    PnP-ADMM with adaptive eta parameter scheduling.
    """
    
    def __init__(self, forward_op, adjoint_op, denoiser, eta=1e-4, 
                 max_iter=50, cg_max_iter=10, cg_tol=1e-6,
                 eta_schedule='exponential', eta_decay_rate=0.95, 
                 eta_min=1e-6):
        """
        Initialize with adaptive eta scheduling.
        
        Args:
            eta_schedule: 'exponential', 'linear', or 'cosine'
            eta_decay_rate: Decay rate for exponential schedule
            eta_min: Minimum value for eta
        """
        super().__init__(forward_op, adjoint_op, denoiser, eta, 
                        max_iter, cg_max_iter, cg_tol)
        
        self.eta_initial = eta
        self.eta_schedule = eta_schedule
        self.eta_decay_rate = eta_decay_rate
        self.eta_min = eta_min
        
    def update_eta(self, iteration):
        """Update eta based on schedule."""
        if self.eta_schedule == 'exponential':
            new_eta = self.eta_initial * (self.eta_decay_rate ** iteration)
        elif self.eta_schedule == 'linear':
            new_eta = self.eta_initial * (1 - iteration / self.max_iter)
        elif self.eta_schedule == 'cosine':
            new_eta = self.eta_min + 0.5 * (self.eta_initial - self.eta_min) * \
                     (1 + np.cos(np.pi * iteration / self.max_iter))
        else:
            new_eta = self.eta_initial
            
        self.eta = max(new_eta, self.eta_min)
        
    def solve_adaptive(self, y_degraded, ground_truth=None, verbose=True):
        """Solve with adaptive eta scheduling."""
        device = y_degraded.device
        
        # Initialize variables (following base class pattern)
        x = torch.zeros_like(y_degraded)
        v = torch.zeros_like(y_degraded)
        u = torch.zeros_like(y_degraded)
        
        mse_history = []
        eta_history = []
        
        for k in range(self.max_iter):
            # Update eta
            self.update_eta(k)
            eta_history.append(self.eta)
            
            # Create system operator with current eta
            system_op = self._create_system_operator(y_degraded.shape, device)
            
            # Standard PnP-ADMM updates
            rhs = self.adjoint_op(y_degraded) + self.eta * (v - u)
            x = conjugate_gradient(system_op, rhs, x.clone(), self.cg_max_iter, self.cg_tol)
            
            v = self.denoiser(x + u)
            u = u + (x - v)
            
            # Clip to valid range
            x = torch.clamp(x, 0, 1)
            
            # Track MSE if ground truth available
            if ground_truth is not None:
                mse_history.append(mse_loss(x, ground_truth))
            
            if verbose and (k + 1) % 10 == 0:
                status = f"Iter {k+1}/{self.max_iter}, η: {self.eta:.2e}"
                if ground_truth is not None:
                    status += f", MSE: {mse_history[-1]:.6f}"
                print(status)
        
        return x, {
            'mse_history': mse_history,
            'eta_history': eta_history
        } 