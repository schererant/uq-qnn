from typing import Callable, Optional
import numpy as np

from data.base import DataSource
from data.synthetic import QuarticData, CustomSyntheticData
from data.measured import MeasuredData

def get_data_source(
    name: str,
    n_data: int = 100,
    sigma_noise: float = 0.0,
    custom_func: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    measured_path: Optional[str] = None,
    seed: int = None
) -> DataSource:
    """
    Factory function to get a data source by name.
    
    Args:
        name (str): Name of the data source ('quartic', 'custom', 'measured')
        n_data (int): Number of data points (for synthetic data)
        sigma_noise (float): Noise level (for synthetic data)
        custom_func (Callable, optional): Custom function for 'custom' data source
        measured_path (str, optional): Path to measured data for 'measured' data source
        seed (int, optional): Seed for reproducibility
        
    Returns:
        DataSource: Instantiated data source
    """
    if name == 'quartic':
        return QuarticData(n_data=n_data, sigma_noise=sigma_noise)
    elif name == 'custom':
        if custom_func is None:
            raise ValueError("custom_func must be provided for 'custom' data source")
        return CustomSyntheticData(custom_func, n_data=n_data, sigma_noise=sigma_noise)
    elif name == 'measured':
        if measured_path is None:
            raise ValueError("measured_path must be provided for 'measured' data source")
        return MeasuredData(measured_path)
    else:
        raise ValueError(f"Unknown data source: {name}") 