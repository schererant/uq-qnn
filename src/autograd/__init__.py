from autograd.base import GradientMethod
from autograd.psr import PSRGradient
from autograd.finite_diff import FiniteDiffGradient


def get_gradient_method(name: str, *args, backend: str = 'numpy', **kwargs) -> GradientMethod:
    """
    Factory function to get a gradient method by name.
    
    Args:
        name (str): Name of the gradient method ('psr', 'finite_diff')
        *args, **kwargs: Arguments for the gradient method constructor
        backend (str): Backend to use ('numpy' or 'torch')
    Returns:
        GradientMethod: Instantiated gradient method
    """
    if name == 'psr':
        return PSRGradient(*args, backend=backend, **kwargs)
    elif name == 'finite_diff':
        return FiniteDiffGradient(*args, backend=backend, **kwargs)
    else:
        raise ValueError(f"Unknown gradient method: {name}")

__all__ = ['GradientMethod', 'PSRGradient', 'FiniteDiffGradient', 'get_gradient_method'] 