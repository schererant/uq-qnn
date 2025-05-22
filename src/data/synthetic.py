import numpy as np
from typing import Tuple, Callable
from data.base import DataSource

class QuarticData(DataSource):
    """
    Synthetic data source generating quartic (x^4) data with optional noise.
    """
    def __init__(self, n_data: int = 100, sigma_noise: float = 0.0):
        """
        Initialize QuarticData.
        
        Args:
            n_data (int): Number of training data points
            sigma_noise (float): Standard deviation of Gaussian noise
        """
        self.n_data = n_data
        self.sigma_noise = sigma_noise

    def _quartic(self, x: np.ndarray) -> np.ndarray:
        """Compute x^4"""
        return np.power(x, 4)

    def get_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        x_min, x_max = 0.0, 1.0
        X = np.linspace(x_min, x_max, self.n_data)
        y = self._quartic(X) + np.random.normal(0, self.sigma_noise, size=self.n_data)
        
        # Create gap in training data
        gap = (x_min + 0.35 * (x_max - x_min), x_min + 0.60 * (x_max - x_min))
        mask = ~((X > gap[0]) & (X < gap[1]))
        X_train, y_train = X[mask], y[mask]
        
        # Generate test data on dense grid
        X_test = np.linspace(x_min, x_max, 500)
        y_test = self._quartic(X_test)
        
        return X_train, y_train, X_test, y_test

class CustomSyntheticData(DataSource):
    """
    Generic synthetic data source with user-provided function.
    """
    def __init__(self, func: Callable[[np.ndarray], np.ndarray], n_data: int = 100, sigma_noise: float = 0.0):
        """
        Initialize CustomSyntheticData.
        
        Args:
            func (Callable): Function to generate y from x
            n_data (int): Number of training data points
            sigma_noise (float): Standard deviation of Gaussian noise
        """
        self.func = func
        self.n_data = n_data
        self.sigma_noise = sigma_noise

    def get_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        x_min, x_max = 0.0, 1.0
        X = np.linspace(x_min, x_max, self.n_data)
        y = self.func(X) + np.random.normal(0, self.sigma_noise, size=self.n_data)
        
        gap = (x_min + 0.35 * (x_max - x_min), x_min + 0.60 * (x_max - x_min))
        mask = ~((X > gap[0]) & (X < gap[1]))
        X_train, y_train = X[mask], y[mask]
        
        X_test = np.linspace(x_min, x_max, 500)
        y_test = self.func(X_test)
        
        return X_train, y_train, X_test, y_test 