import numpy as np
from utils.helpers import load_model_params
from typing import Any

def run_inference(
    simulation_runner: Any,
    params: np.ndarray,
    X: np.ndarray,
    n_samples: int,
    data_source: Any
) -> np.ndarray:
    """
    Run inference using the simulation runner and given parameters.
    Args:
        simulation_runner: The simulation runner instance
        params (np.ndarray): Model parameters (array)
        X (np.ndarray): Input data
        n_samples (int): Number of samples for simulation
        data_source: Data source object with encode_phase method
    Returns:
        np.ndarray: Predicted values
    """
    enc_X = data_source.encode_phase(X)
    preds = simulation_runner.run_sequence(params, enc_X, n_samples)
    return preds


def run_inference_from_file(
    simulation_runner: Any,
    param_path: str,
    X: np.ndarray,
    n_samples: int,
    data_source: Any
) -> np.ndarray:
    params = load_model_params(param_path)
    return run_inference(simulation_runner, params, X, n_samples, data_source) 