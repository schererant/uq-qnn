from autograd.base import GradientMethod
from autograd.psr import PSRGradient
from autograd.finite_diff import FiniteDiffGradient
from .psr_torch import MemristorLossPSR, photonic_psr_coeffs_torch

def get_gradient_method(name: str, *args, **kwargs) -> GradientMethod:
    """
    Factory function to get a gradient method by name.
    
    Args:
        name (str): Name of the gradient method ('psr', 'finite_diff')
        *args, **kwargs: Arguments for the gradient method constructor
        
    Returns:
        GradientMethod: Instantiated gradient method
    """
    if name == 'psr':
        return PSRGradient(*args, **kwargs)
    elif name == 'finite_diff':
        return FiniteDiffGradient(*args, **kwargs)
    else:
        raise ValueError(f"Unknown gradient method: {name}")

__all__ = ['GradientMethod', 'PSRGradient', 'FiniteDiffGradient', 'get_gradient_method'] 