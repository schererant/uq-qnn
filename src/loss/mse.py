import numpy as np
from loss.base import LossFunction

class MSELoss(LossFunction):
    """
    Mean squared error loss function.
    """
    def compute(self, preds: np.ndarray, targets: np.ndarray) -> float:
        """
        Compute the mean squared error loss.
        
        Args:
            preds (np.ndarray): Predicted values
            targets (np.ndarray): Target values
        Returns:
            float: MSE loss value
        """
        return 0.5 * np.mean((preds - targets) ** 2) 