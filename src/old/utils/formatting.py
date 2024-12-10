import tensorflow as tf
import numpy as np


def format_metrics(metrics_dict, indent=0):
    """Format metrics dictionary into lines with proper indentation"""
    lines = []
    indent_str = " " * indent
    for key, value in metrics_dict.items():
        if isinstance(value, tf.Variable):
            # Extract numeric value from TensorFlow variable
            value = float(value.numpy())
            lines.append(f"{indent_str}{key}: {value:.4f}")
        elif isinstance(value, dict):
            lines.append(f"{indent_str}{key}:")
            lines.extend(format_metrics(value, indent + 2))
        elif isinstance(value, (np.ndarray, list)):
            lines.append(f"{indent_str}{key}: {np.mean(value):.4f}")
        else:
            lines.append(f"{indent_str}{key}: {value}")
    return lines

def format_hyperparameters(hyperparameters_dict, indent=0):
    lines = []
    indent_str = " " * indent
    for key, value in hyperparameters_dict.items():
        if isinstance(value, float):
            lines.append(f"{indent_str}{key}: {value:.6f}")
        else:
            lines.append(f"{indent_str}{key}: {value}")
    return lines

