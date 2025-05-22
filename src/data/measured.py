import pickle
import numpy as np
from typing import Tuple
from data.base import DataSource

class MeasuredData(DataSource):
    """
    Data source for loading measured data from pickle files.
    """
    def __init__(self, path: str):
        """
        Initialize MeasuredData.
        
        Args:
            path (str): Path to pickle file containing (X, y) data
        """
        self.path = path

    def load_pickle(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load data from pickle file.
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: X and y arrays
        """
        with open(self.path, "rb") as fh:
            X, y = pickle.load(fh)
        return np.asarray(X), np.asarray(y)

    def get_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        X_train, y_train = self.load_pickle()
        
        # For measured data, we create a synthetic test set that densifies
        # the full phase range, using quartic as placeholder when ground truth unknown
        X_test = np.linspace(0.0, 1.0, 500)
        y_test = np.power(X_test, 4)  # Placeholder
        
        return X_train, y_train, X_test, y_test 