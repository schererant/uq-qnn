from loss.base import LossFunction
from loss.mse import MSELoss
from loss.custom_loss import CustomLoss

def get_loss_function(name: str, *args, **kwargs) -> LossFunction:
    """
    Factory function to get a loss function by name.
    
    Args:
        name (str): Name of the loss function ('mse', 'custom')
        *args, **kwargs: Arguments for the loss function constructor
        
    Returns:
        LossFunction: Instantiated loss function
    """
    if name == 'mse':
        return MSELoss(*args, **kwargs)
    elif name == 'custom':
        return CustomLoss(*args, **kwargs)
    else:
        raise ValueError(f"Unknown loss function: {name}")

__all__ = ['LossFunction', 'MSELoss', 'CustomLoss', 'get_loss_function'] 