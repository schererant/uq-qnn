from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple

class DataSource(ABC):
    @abstractmethod
    def get_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Get training and test data.
        
        Returns:
            Tuple containing:
                X_train (np.ndarray): Training input data
                y_train (np.ndarray): Training target data
                X_test (np.ndarray): Test input data
                y_test (np.ndarray): Test target data
        """
        pass

    def encode_phase(self, X: np.ndarray) -> np.ndarray:
        """
        Encode input data into phase values.
        
        Args:
            X (np.ndarray): Input data in [0,1]
            
        Returns:
            np.ndarray: Encoded phases as 2*arccos(X)
        """
        return 2 * np.arccos(X) 