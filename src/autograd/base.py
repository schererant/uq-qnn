from abc import ABC, abstractmethod
import numpy as np
from typing import Any, Tuple

class GradientMethod(ABC):
    """
    Abstract base class for gradient computation methods.
    """
    @abstractmethod
    def compute_gradients(self, 
                         params: np.ndarray, 
                         *args, **kwargs) -> np.ndarray:
        """
        Compute gradients for the given parameters.
        
        Args:
            params (np.ndarray): Model parameters
            *args, **kwargs: Additional arguments as needed
        Returns:
            np.ndarray: Gradient vector
        """
        pass 