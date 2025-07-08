import torch
from typing import Callable

def conjugate_gradient(A: Callable[[torch.Tensor], torch.Tensor], 
                      b: torch.Tensor, 
                      x0: torch.Tensor, 
                      max_iter: int = 100, 
                      tol: float = 1e-6) -> torch.Tensor:
    """
    Conjugate Gradient method for solving Ax=b.
    
    Args:
        A: Function that returns Ax for input x
        b: Right-hand side vector
        x0: Initial guess
        max_iter: Maximum number of iterations
        tol: Tolerance for convergence
        
    Returns:
        Solution x
    """
    x = x0.clone()
    r = b - A(x)
    d = r.clone()
    
    for _ in range(max_iter):
        z = A(d)
        rr = torch.sum(r**2)
        alpha = rr / torch.sum(d * z)
        x += alpha * d
        r -= alpha * z
        
        if torch.norm(r) / torch.norm(b) < tol:
            break
            
        beta = torch.sum(r**2) / rr
        d = r + beta * d
        
    return x 