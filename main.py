#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Main entry point for UQ-QNN: Uncertainty Quantification for Quantum Neural Networks

This script provides a command-line interface to run the photonic neural network
training pipeline with various configuration options.
"""

import argparse
import time
import numpy as np
import matplotlib.pyplot as plt

from src.data import get_data, load_measurement_pickle, quartic_data
from src.training import train_pytorch
from src.simulation import run_simulation_sequence_np, sim_logger
from src.utils import config, _resolve_n_swipe
from src.data import get_cont_swipe_data


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="UQ-QNN: Photonic Neural Network Training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data options
    data_group = parser.add_argument_group('Data')
    data_group.add_argument("--measured-data", type=str, default=None,
                           help="Path to measured data pickle file (X,y)")
    data_group.add_argument("--n-data", type=int, default=100,
                           help="Number of synthetic data points to generate")
    data_group.add_argument("--sigma-noise", type=float, default=0.1,
                           help="Standard deviation of noise for synthetic data")
    data_group.add_argument("--datafunction", type=str, default='quartic_data',
                           choices=['quartic_data', 'sinusoid_data', 'multi_modal_data', 
                                    'step_function_data', 'oscillating_poly_data', 'damped_cosine_data'],
                           help="Synthetic data function to use")
    
    # Simulation options
    sim_group = parser.add_argument_group('Simulation')
    sim_group.add_argument("--n-samples", type=int, default=1000,
                          help="Number of samples for circuit simulation")
    sim_group.add_argument("--memory-depth", type=int, default=2,
                          help="Memory buffer depth for memristor circuit")
    
    # Continuous swipe options
    swipe_group = parser.add_argument_group('Continuous Swipe')
    swipe_group.add_argument("--continuous", action="store_true",
                            help="Use continuous-swipe training")
    swipe_group.add_argument("--n-swipe", type=int, default=None,
                            help="Number of phase points per data point")
    swipe_group.add_argument("--swipe-span", type=float, default=np.pi/20,
                            help="Total phase span for swiping in radians")
    swipe_group.add_argument("--t-phase-ms", type=float, default=10.0,
                            help="Heater settle time in milliseconds")
    swipe_group.add_argument("--f-laser-khz", type=float, default=50.0,
                            help="Laser repetition rate in kHz")
    swipe_group.add_argument("--det-window-us", type=float, default=10.0,
                            help="Detector integration window in microseconds")
    swipe_group.add_argument("--max-swipe", type=int, default=21,
                            help="Maximum allowed swipe count")
    
    # Training options
    train_group = parser.add_argument_group('Training')
    train_group.add_argument("--lr", type=float, default=0.03,
                            help="Learning rate")
    train_group.add_argument("--epochs", type=int, default=50,
                            help="Number of training epochs")
    train_group.add_argument("--seed", type=int, default=42,
                            help="Random seed for reproducibility")
    
    # Output options
    output_group = parser.add_argument_group('Output')
    output_group.add_argument("--no-plot", action="store_true",
                             help="Disable plotting")
    output_group.add_argument("--save-model", type=str, default=None,
                             help="Path to save trained model parameters")
    output_group.add_argument("--save-plot", type=str, default=None,
                             help="Path to save plots (without extension)")
    
    return parser.parse_args()


def update_config_from_args(args):
    """Update the configuration dictionary from command line arguments."""
    # Data settings
    config['n_data'] = args.n_data
    config['sigma_noise'] = args.sigma_noise
    config['datafunction'] = args.datafunction
    config['memory_depth'] = args.memory_depth
    
    # Timing settings
    config['t_phase_ms'] = args.t_phase_ms
    config['f_laser_khz'] = args.f_laser_khz
    config['det_window_us'] = args.det_window_us
    config['max_swipe'] = args.max_swipe
    
    # Continuous swipe settings
    config['use_continuous'] = args.continuous
    config['n_swipe'] = args.n_swipe
    config['swipe_span'] = args.swipe_span
    
    # Training settings
    config['lr'] = args.lr
    config['epochs'] = args.epochs
    
    # Output settings
    config['do_plot'] = not args.no_plot
    
    # Sampler settings
    config['n_samples'] = args.n_samples


def run_training(X_train, y_train, X_test, y_test, args):
    """Run the training process with the given data."""
    start_time = time.time()
    
    # Determine continuous or discrete mode
    continuous_mode = args.continuous
    
    if continuous_mode:
        n_swipe = _resolve_n_swipe()
        config['n_swipe'] = n_swipe  # freeze for the rest of the run
        print(f"Running in continuous mode with n_swipe={n_swipe}")
        theta_opt, history = train_pytorch(
            X_train, y_train,
            memory_depth=config['memory_depth'],
            lr=config['lr'],
            epochs=config['epochs'],
            phase_idx=config['phase_idx'],
            n_photons=config['n_photons'],
            n_swipe=config['n_swipe'],
            swipe_span=config['swipe_span'],
            n_samples=args.n_samples,
            seed=args.seed
        )
    else:
        print("Running in discrete mode")
        theta_opt, history = train_pytorch(
            X_train, y_train,
            memory_depth=config['memory_depth'],
            lr=config['lr'],
            epochs=config['epochs'],
            phase_idx=config['phase_idx'],
            n_photons=config['n_photons'],
            n_swipe=0,
            swipe_span=0.0,
            n_samples=args.n_samples,
            seed=args.seed
        )
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    print("Optimized parameters:", theta_opt)
    
    # Generate predictions on test data
    if continuous_mode:
        preds = run_simulation_sequence_np(
            theta_opt, config['memory_depth'], args.n_samples,
            encoded_phases=2 * np.arccos(X_test), n_swipe=config['n_swipe'], swipe_span=config['swipe_span']
        )
    else:
        enc_test = 2 * np.arccos(X_test)
        preds = run_simulation_sequence_np(
            theta_opt, config['memory_depth'], args.n_samples,
            encoded_phases=enc_test
        )
    
    # Calculate mean squared error on test data
    mse = np.mean((preds - y_test) ** 2)
    print(f"Test MSE: {mse:.6f}")
    
    # Save model if requested
    if args.save_model:
        np.save(args.save_model, theta_opt)
        print(f"Saved model parameters to {args.save_model}")
    
    # Plot results if enabled
    if config['do_plot']:
        plot_results(X_train, y_train, X_test, y_test, preds, history, continuous_mode, args)
    
    return theta_opt, history, preds


def plot_results(X_train, y_train, X_test, y_test, preds, history, continuous_mode, args):
    """Plot training results."""
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(18, 10))
    
    # Plot 1: Loss curve
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(history)
    ax1.set_yscale('log')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.grid(True)
    
    # Plot 2: Data and model fit
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.scatter(X_train, y_train, s=20, label='Training data', zorder=3)
    
    if continuous_mode:
        enc_swipe, _ = get_cont_swipe_data(
            X_train, y_train, 
            n_swipe=config['n_swipe'], 
            swipe_span=config['swipe_span']
        )
        X_swipe = np.cos(enc_swipe / 2)
        ax2.scatter(
            X_swipe, 
            np.repeat(y_train, config['n_swipe']), 
            s=8, alpha=0.35, 
            label=f'Swipe (n={config["n_swipe"]})', 
            zorder=2
        )
    
    ax2.plot(X_test, y_test, label='Ground truth', ls='--', zorder=1)
    ax2.plot(X_test, preds, label='Model prediction', c='red', zorder=4)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('Model Fit')
    ax2.grid(True)
    ax2.legend()
    
    # Plot 3: Phase encoding visualization
    ax3 = fig.add_subplot(2, 2, 3)
    idx = np.arange(len(X_train))
    enc_orig = 2 * np.arccos(X_train)
    ax3.plot(idx, enc_orig, '-o', label='Original encoded phases', lw=1.5)
    
    if continuous_mode:
        idx_swipe = np.repeat(idx, config['n_swipe'])
        ax3.scatter(idx_swipe, enc_swipe, s=6, alpha=0.35, label='Swipe encoded phases')
    
    ax3.set_xlabel('Data index')
    ax3.set_ylabel('Encoding phase [rad]')
    ax3.set_title('Phase Encoding')
    ax3.grid(True)
    ax3.legend()
    
    # Plot 4: Residuals
    ax4 = fig.add_subplot(2, 2, 4)
    residuals = preds - y_test
    ax4.scatter(X_test, residuals)
    ax4.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    ax4.set_xlabel('x')
    ax4.set_ylabel('Residual (prediction - truth)')
    ax4.set_title('Prediction Residuals')
    ax4.grid(True)
    
    plt.tight_layout()
    
    # Save plots if requested
    if args.save_plot:
        plt.savefig(f"{args.save_plot}.png", dpi=300)
        plt.savefig(f"{args.save_plot}.pdf")
        print(f"Saved plots to {args.save_plot}.png and {args.save_plot}.pdf")
    
    plt.show()


def main():
    """Main entry point."""
    args = parse_arguments()
    update_config_from_args(args)
    
    print("=== UQ-QNN: Photonic Neural Network Training ===")
    print(f"Running with configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Load or generate data
    if args.measured_data:
        print(f"Loading measured data from {args.measured_data}")
        X_train, y_train = load_measurement_pickle(args.measured_data)
        # Generate synthetic test set for evaluation
        X_test = np.linspace(0.0, 1.0, 500)
        
        # Use the specified data function for test set
        from src.data import quartic_data, sinusoid_data, multi_modal_data, step_function_data, oscillating_poly_data, damped_cosine_data
        data_funcs = {
            'quartic_data': quartic_data,
            'sinusoid_data': sinusoid_data,
            'multi_modal_data': multi_modal_data,
            'step_function_data': step_function_data,
            'oscillating_poly_data': oscillating_poly_data,
            'damped_cosine_data': damped_cosine_data
        }
        y_test = data_funcs[config['datafunction']](X_test)
    else:
        print(f"Generating synthetic data with {args.n_data} points and noise σ={args.sigma_noise}")
        print(f"Using data function: {config['datafunction']}")
        X_train, y_train, X_test, y_test = get_data(
            config['n_data'],
            config['sigma_noise'],
            config['datafunction']
        )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Run the training
    theta_opt, history, preds = run_training(X_train, y_train, X_test, y_test, args)
    
    # Print simulation statistics
    sim_logger.report()
    
    return 0


if __name__ == "__main__":
    main()