from abc import ABC, abstractmethod
import numpy as np

class LossFunction(ABC):
    """
    Abstract base class for loss functions.
    """
    @abstractmethod
    def compute(self, preds: np.ndarray, targets: np.ndarray) -> float:
        """
        Compute the loss value.
        
        Args:
            preds (np.ndarray): Predicted values
            targets (np.ndarray): Target values
        Returns:
            float: Loss value
        """
        pass 