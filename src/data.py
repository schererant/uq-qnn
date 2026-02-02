from __future__ import annotations

import pickle
from typing import Tuple
import numpy as np


def load_measurement_pickle(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads measured data from a pickle file and returns it as two numpy arrays.
    Args:
        path (str): Path to the pickle file containing (X, y) data.
    Returns:
        Tuple[np.ndarray, np.ndarray]: Tuple of X and y arrays.
    """
    with open(path, "rb") as fh:
        X, y = pickle.load(fh)
    return np.asarray(X), np.asarray(y)


def compute_n_swipe(
    t_phase_ms: float,
    f_laser_khz: float,
    det_window_us: float,
    max_swipe: int = 201,
) -> int:
    """
    Translates hardware-timing limits into a safe, odd swipe count. 
    It divides the heater's settle time by the slower of the laser 
    period and detector window to see how many optical "slots" fit, 
    then forces the result to be odd (so the original point stays centered) 
    and caps it at `max_swipe` to keep memory footprint reasonable.
    Args:
        t_phase_ms (float): Heater settle time in milliseconds.
        f_laser_khz (float): Laser repetition rate in kHz.
        det_window_us (float): Detector integration window in microseconds.
        max_swipe (int): Maximum allowed swipe count.
    Returns:
        int: Odd integer swipe count, capped at max_swipe.
    """
    if t_phase_ms <= 0 or f_laser_khz <= 0 or det_window_us <= 0:
        raise ValueError("Timing inputs must be positive.")

    period_laser_us = 1_000 / f_laser_khz  # µs
    slot_us = max(period_laser_us, det_window_us)
    slots_total = int((t_phase_ms * 1_000) // slot_us)  # integer slots
    n_swipe = max(1, 2 * (slots_total // 2) + 1)        # force odd
    return min(n_swipe, max_swipe)


def get_cont_swipe_data(
    X: np.ndarray,
    y: np.ndarray,
    n_swipe: int,
    swipe_span: float = np.pi / 20,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Expands each (X[i], y[i]) into n_swipe neighboring phase points for data augmentation.
    Args:
        X (np.ndarray): Input data array.
        y (np.ndarray): Output data array.
        n_swipe (int): Number of phase points to generate per data point (must be odd).
        swipe_span (float): Total phase span for swiping.
    Returns:
        Tuple[np.ndarray, np.ndarray]: Expanded phase-encoded X and repeated y arrays.
    """
    if n_swipe < 1 or n_swipe % 2 == 0:
        raise ValueError(f"n_swipe must be a positive odd integer (got {n_swipe})")
    enc_base = 2 * np.arccos(X)
    offsets = np.linspace(-swipe_span / 2, swipe_span / 2, n_swipe, dtype=enc_base.dtype)
    enc_swipe = np.concatenate([enc + offsets for enc in enc_base])
    y_swipe = np.repeat(y, n_swipe)
    return enc_swipe, y_swipe


def get_measured_swipe_data(
    encoded_phases: np.ndarray,
    measured_phases: np.ndarray,
    n_swipe: int
) -> np.ndarray:
    """
    For each encoded phase, find the closest measured phase and select a window of n_swipe measured phases
    centered around it. Handles edge cases at the boundaries.
    Args:
        encoded_phases (np.ndarray): 1D array of encoded phases (e.g., 2*arccos(X)).
        measured_phases (np.ndarray): 1D array of measured phase values.
        n_swipe (int): Number of phase points per data point (should be odd).
    Returns:
        np.ndarray: 2D array of shape (len(encoded_phases), n_swipe) with measured phase values for each swipe.
    """
    if n_swipe < 1 or n_swipe % 2 == 0:
        raise ValueError(f"n_swipe must be a positive odd integer (got {n_swipe})")
    n_data = len(encoded_phases)
    n_meas = len(measured_phases)
    enc_samples = np.empty((n_data, n_swipe), dtype=measured_phases.dtype)
    n2 = n_swipe // 2
    for i in range(n_data):
        # Find the index of the measured phase closest to the encoded phase
        diffs = np.abs(measured_phases - encoded_phases[i])
        k = np.argmin(diffs)
        # Compute window bounds, handle boundaries
        start = max(0, k - n2)
        end = min(n_meas, k + n2 + 1)
        # If at the left edge, pad right
        if end - start < n_swipe:
            if start == 0:
                end = min(n_meas, start + n_swipe)
            else:
                start = max(0, end - n_swipe)
        # Fill the row
        window = measured_phases[start:end]
        # If window is too short (at edges), pad with edge values
        if len(window) < n_swipe:
            if start == 0:
                window = np.pad(window, (0, n_swipe - len(window)), mode='edge')
            else:
                window = np.pad(window, (n_swipe - len(window), 0), mode='edge')
        enc_samples[i, :] = window
    return enc_samples


def quartic_data(x: np.ndarray) -> np.ndarray:
    """
    Computes the quartic (x^4) of the input array.
    Args:
        x (np.ndarray): Input array.
    Returns:
        np.ndarray: Output array with each element raised to the 4th power.
    """
    return np.power(x, 4)


def sinusoid_data(x: np.ndarray) -> np.ndarray:
    """
    Computes a sinusoidal function of the input array.
    Args:
        x (np.ndarray): Input array.
    Returns:
        np.ndarray: Output array with sin(2πx) * 0.5 + 0.5
    """
    return np.sin(2 * np.pi * x) * 0.5 + 0.5


def multi_modal_data(x: np.ndarray) -> np.ndarray:
    """
    Computes a multi-modal function (sum of Gaussians) of the input array.
    Args:
        x (np.ndarray): Input array.
    Returns:
        np.ndarray: Output array with multiple Gaussian peaks
    """
    g1 = np.exp(-((x - 0.2) ** 2) / 0.02) * 0.5
    g2 = np.exp(-((x - 0.6) ** 2) / 0.04) * 0.8
    g3 = np.exp(-((x - 0.9) ** 2) / 0.01) * 0.3
    return g1 + g2 + g3


def step_function_data(x: np.ndarray) -> np.ndarray:
    """
    Computes a step function with smooth transitions.
    Args:
        x (np.ndarray): Input array.
    Returns:
        np.ndarray: Output array with step function values
    """
    return 0.5 * (np.tanh((x - 0.3) * 10) + 1) * 0.5 + 0.25


def oscillating_poly_data(x: np.ndarray) -> np.ndarray:
    """
    Computes an oscillating polynomial function.
    Args:
        x (np.ndarray): Input array.
    Returns:
        np.ndarray: Output array with oscillating polynomial values
    """
    return x**3 - 0.5 * x**2 + 0.1 * np.sin(15 * x)


def damped_cosine_data(x: np.ndarray) -> np.ndarray:
    """
    Computes a damped cosine function.
    Args:
        x (np.ndarray): Input array.
    Returns:
        np.ndarray: Output array with damped cosine values
    """
    return np.exp(-2 * x) * np.cos(10 * np.pi * x) * 0.5 + 0.5


def one_hot_encode(labels: np.ndarray, n_classes: int) -> np.ndarray:
    """
    Convert integer class labels to one-hot encoding.
    Args:
        labels (np.ndarray): Integer labels of shape (n_samples,).
        n_classes (int): Number of classes.
    Returns:
        np.ndarray: One-hot encoded labels of shape (n_samples, n_classes).
    """
    n_samples = len(labels)
    one_hot = np.zeros((n_samples, n_classes))
    one_hot[np.arange(n_samples), labels.astype(int)] = 1.0
    return one_hot


def generate_classification_data(
    n_data: int = 100,
    n_classes: int = 2,
    data_type: str = 'binary_threshold',
    noise_level: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic classification datasets.
    Args:
        n_data (int): Number of data points.
        n_classes (int): Number of classes (2 for binary, 3+ for multi-class).
        data_type (str): Type of classification data:
            - 'binary_threshold': Simple threshold at x=0.5
            - 'multi_class_regions': Three regions [0,0.33], [0.33,0.66], [0.66,1.0]
            - 'sinusoidal': Classes based on sin(2πx) sign
        noise_level (float): Probability of flipping labels (for noise).
    Returns:
        Tuple[np.ndarray, np.ndarray]: (X, y) where y is integer labels.
    """
    X = np.linspace(0.0, 1.0, n_data)
    
    if data_type == 'binary_threshold':
        if n_classes != 2:
            raise ValueError("binary_threshold only supports 2 classes")
        y = (X > 0.5).astype(int)
    elif data_type == 'multi_class_regions':
        y = np.zeros(n_data, dtype=int)
        y[(X > 0.33) & (X <= 0.66)] = 1
        y[X > 0.66] = 2
        if n_classes != 3:
            raise ValueError("multi_class_regions only supports 3 classes")
    elif data_type == 'sinusoidal':
        if n_classes != 2:
            raise ValueError("sinusoidal only supports 2 classes")
        y = (np.sin(2 * np.pi * X) > 0).astype(int)
    else:
        raise ValueError(f"Unknown data_type: {data_type}")
    
    # Add noise by randomly flipping labels
    if noise_level > 0:
        flip_mask = np.random.random(n_data) < noise_level
        if n_classes == 2:
            y[flip_mask] = 1 - y[flip_mask]
        else:
            # For multi-class, randomly assign to different class
            for i in np.where(flip_mask)[0]:
                other_classes = [c for c in range(n_classes) if c != y[i]]
                y[i] = np.random.choice(other_classes)
    
    return X, y


def get_classification_data(
    n_data: int = 100,
    n_classes: int = 2,
    data_type: str = 'binary_threshold',
    noise_level: float = 0.0,
    return_one_hot: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates synthetic training and test data for classification tasks.
    Args:
        n_data (int): Number of training data points.
        n_classes (int): Number of classes.
        data_type (str): Type of classification data (see generate_classification_data).
        noise_level (float): Probability of label noise.
        return_one_hot (bool): If True, return one-hot encoded labels.
    Returns:
        Tuple: (X_train, y_train, X_test, y_test) arrays.
            If return_one_hot=True, y arrays are one-hot encoded.
    """
    x_min, x_max = 0.0, 1.0
    X = np.linspace(x_min, x_max, n_data)
    y = generate_classification_data(n_data, n_classes, data_type, noise_level)[1]
    
    # Create train/test split with gap (similar to regression)
    gap = (x_min + 0.35 * (x_max - x_min), x_min + 0.60 * (x_max - x_min))
    mask = ~((X > gap[0]) & (X < gap[1]))
    X_train, y_train = X[mask], y[mask]
    
    # Generate test data
    X_test = np.linspace(x_min, x_max, 500)
    y_test = generate_classification_data(500, n_classes, data_type, 0.0)[1]
    
    if return_one_hot:
        y_train = one_hot_encode(y_train, n_classes)
        y_test = one_hot_encode(y_test, n_classes)
    
    return X_train, y_train, X_test, y_test


def get_data(
    n_data: int = 100,
    sigma_noise: float = 0.0,
    datafunction: str = 'quartic_data',
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates synthetic training and test data, with a gap in the training set.
    Args:
        n_data (int): Number of training data points.
        sigma_noise (float): Standard deviation of Gaussian noise added to y.
        datafunction (str): Name of the function to generate y from X.
    Returns:
        Tuple: (X_train, y_train, X_test, y_test) arrays.
    """
    x_min, x_max = 0.0, 1.0
    X = np.linspace(x_min, x_max, n_data)
    # Explicit mapping of string to function
    datafunction_map = {
        'quartic_data': quartic_data,
        'sinusoid_data': sinusoid_data,
        'multi_modal_data': multi_modal_data,
        'step_function_data': step_function_data,
        'oscillating_poly_data': oscillating_poly_data,
        'damped_cosine_data': damped_cosine_data,
    }
    if datafunction not in datafunction_map:
        raise ValueError(f"Unknown datafunction: {datafunction}. Available functions: {list(datafunction_map.keys())}")
    datafunc = datafunction_map[datafunction]
    y = datafunc(X) + np.random.normal(0, sigma_noise, size=n_data)
    gap = (x_min + 0.35 * (x_max - x_min), x_min + 0.60 * (x_max - x_min))
    mask = ~((X > gap[0]) & (X < gap[1]))
    X_train, y_train = X[mask], y[mask]
    X_test = np.linspace(x_min, x_max, 500)
    y_test = datafunc(X_test)
    return X_train, y_train, X_test, y_test