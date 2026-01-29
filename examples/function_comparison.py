#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Function comparison example using the UQ-QNN framework.

This example demonstrates training and evaluating the photonic neural network
on different synthetic functions to compare performance and uncertainty.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import get_data
from src.training import train_pytorch
from src.simulation import run_simulation_sequence_np, sim_logger
from src.utils import config


def train_and_evaluate(datafunction, n_data=80, sigma_noise=0.05, n_samples=500, epochs=30, n_phases=2):
    """
    Train and evaluate a model on the specified data function.

    Args:
        datafunction (str): Name of the data function to use
        n_data (int): Number of data points to generate
        sigma_noise (float): Noise level to add to data
        n_samples (int): Number of samples for circuit simulation
        epochs (int): Number of training epochs

    Returns:
        tuple: (X_train, y_train, X_test, y_test, mean_preds, std_preds, theta_opt, history)
    """
    # Generate synthetic data
    X_train, y_train, X_test, y_test = get_data(
        n_data, sigma_noise, datafunction
    )

    # Train the model
    theta_opt, history = train_pytorch(
        X_train, y_train,
        memory_depth=config['memory_depth'],
        lr=config['lr'],
        epochs=epochs,
        phase_idx=config['phase_idx'],
        n_photons=config['n_photons'],
        n_swipe=0,
        n_samples=n_samples,
        n_phases=n_phases
    )

    # Uncertainty estimation through multiple forward passes
    n_forward_passes = 10
    all_preds = np.zeros((len(X_test), n_forward_passes))

    for i in range(n_forward_passes):
        # Each forward pass with a different sample count introduces some randomness
        sample_count = n_samples + np.random.randint(-100, 100)
        sample_count = max(100, sample_count)  # Ensure at least 100 samples
        
        # Small random perturbation to parameters to simulate quantum noise
        perturbed_theta = theta_opt.copy()
        # Only perturb phases slightly, not the weight
        perturbed_theta[:-1] += np.random.normal(0, 0.05, size=len(perturbed_theta)-1)
        
        enc_test = 2 * np.arccos(X_test)
        preds = run_simulation_sequence_np(
            perturbed_theta, 
            config['memory_depth'], 
            sample_count, 
            encoded_phases=enc_test
        )
        all_preds[:, i] = preds

    # Compute mean and standard deviation of predictions
    mean_preds = np.mean(all_preds, axis=1)
    std_preds = np.std(all_preds, axis=1)

    return X_train, y_train, X_test, y_test, mean_preds, std_preds, theta_opt, history


def compute_metrics(y_test, mean_preds, std_preds):
    """Compute evaluation metrics."""
    mse = np.mean((mean_preds - y_test) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(mean_preds - y_test))

    # Compute normalized uncertainty score (lower is better - well calibrated)
    abs_error = np.abs(mean_preds - y_test)
    uncertainty_score = np.mean(abs_error / (std_preds + 1e-6))

    # Compute prediction interval coverage (should be close to 0.95 for well-calibrated model)
    inside_interval = (y_test >= mean_preds - 2*std_preds) & (y_test <= mean_preds + 2*std_preds)
    coverage = np.mean(inside_interval)

    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'uncertainty_score': uncertainty_score,
        'coverage': coverage
    }


def plot_function_comparison(results, functions):
    """Plot comparison of model performance on different functions."""
    n_functions = len(functions)
    fig, axes = plt.subplots(n_functions, 3, figsize=(18, 4*n_functions))

    if n_functions == 1:
        axes = axes.reshape(1, 3)

    metrics_table = []

    for i, func_name in enumerate(functions):
        result = results[func_name]
        X_train, y_train, X_test, y_test = result['data']
        mean_preds, std_preds = result['predictions']
        history = result['history']
        metrics = result['metrics']

        # Plot training data and predictions
        ax = axes[i, 0]
        ax.scatter(X_train, y_train, s=20, label='Training data', alpha=0.7)
        ax.plot(X_test, y_test, 'k--', label='Ground truth')
        ax.plot(X_test, mean_preds, 'r-', label='Mean prediction')
        ax.fill_between(
            X_test,
            mean_preds - 2*std_preds,
            mean_preds + 2*std_preds,
            color='r', alpha=0.3,
            label='95% confidence interval'
        )
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f'{func_name} - Predictions')
        ax.legend(loc='upper left')
        ax.grid(True)

        # Plot training loss
        ax = axes[i, 1]
        ax.plot(history)
        ax.set_yscale('log')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title(f'{func_name} - Training Loss')
        ax.grid(True)

        # Plot uncertainty vs error
        ax = axes[i, 2]
        abs_error = np.abs(mean_preds - y_test)
        scatter = ax.scatter(std_preds, abs_error, alpha=0.7, c=X_test, cmap='viridis')
        plt.colorbar(scatter, ax=ax, label='x value')
        max_val = max(np.max(std_preds), np.max(abs_error))
        ax.plot([0, max_val], [0, max_val], 'k--', label='y=x (perfect calibration)')
        ax.set_xlabel('Uncertainty (std)')
        ax.set_ylabel('Absolute error')
        ax.set_title(f'{func_name} - Calibration')
        ax.grid(True)
        ax.legend()

        # Collect metrics for table
        metrics_table.append([
            func_name,
            f"{metrics['rmse']:.4f}",
            f"{metrics['mae']:.4f}",
            f"{metrics['uncertainty_score']:.4f}",
            f"{metrics['coverage']:.4f}"
        ])

    plt.tight_layout()

    # Create metrics table as a separate figure
    fig_table, ax_table = plt.subplots(figsize=(10, n_functions*0.5 + 1))
    ax_table.axis('off')
    table = ax_table.table(
        cellText=metrics_table,
        colLabels=['Function', 'RMSE', 'MAE', 'Uncert. Score', 'Coverage'],
        cellLoc='center',
        loc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    ax_table.set_title('Performance Metrics Comparison')

    plt.tight_layout()

    plt.figure(fig.number)  # Return to the main figure

    return fig, fig_table


def main():
    """Main function to run the comparison."""
    print("=== UQ-QNN: Function Comparison Example ===")

    # Set random seed for reproducibility
    np.random.seed(42)

    # Configure parameters
    config['lr'] = 0.03
    config['memory_depth'] = 2
    config['phase_idx'] = (0, 1)
    config['n_photons'] = (1, 1)
    n_phases = 2  # Number of external phase parameters (excluding memory phase)

    # Functions to compare
    functions = [
        'quartic_data', 
        'sinusoid_data', 
        'multi_modal_data',
        'step_function_data',
        'oscillating_poly_data',
        'damped_cosine_data'
    ]
    
    # Print parameter structure info
    print(f"Using array-based circuit structure with {n_phases} phase parameters (+ memory phase)")

    # Parameters
    n_data = 80
    sigma_noise = 0.05
    n_samples = 500
    epochs = 60

    # Store results
    results = {}

    # Train and evaluate on each function
    for func_name in functions:
        print(f"\n=== Training on {func_name} ===")
        X_train, y_train, X_test, y_test, mean_preds, std_preds, theta_opt, history = train_and_evaluate(
            func_name, n_data, sigma_noise, n_samples, epochs, n_phases
        )

        metrics = compute_metrics(y_test, mean_preds, std_preds)
        print(f"  RMSE: {metrics['rmse']:.6f}")
        print(f"  MAE: {metrics['mae']:.6f}")
        print(f"  Uncertainty Score: {metrics['uncertainty_score']:.6f}")
        print(f"  Coverage (95% interval): {metrics['coverage']:.6f}")

        results[func_name] = {
            'data': (X_train, y_train, X_test, y_test),
            'predictions': (mean_preds, std_preds),
            'theta': theta_opt,
            'history': history,
            'metrics': metrics
        }

    # Plot comparison
    fig, fig_table = plot_function_comparison(results, functions)

    # Save figures
    fig.savefig('function_comparison.png', dpi=300, bbox_inches='tight')
    fig_table.savefig('function_metrics.png', dpi=300, bbox_inches='tight')

    plt.show()

    # Print simulation statistics
    sim_logger.report()


if __name__ == "__main__":
    main()
