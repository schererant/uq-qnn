from __future__ import annotations

import argparse
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt

from .data import get_data, load_measurement_pickle, quartic_data, get_cont_swipe_data, compute_n_swipe
from .training import train_pytorch
from .simulation import run_simulation_sequence_np, sim_logger

# ===================== CONFIGURATION =====================
config = {
    # Data generation
    'n_data': 100,
    'sigma_noise': 0.1,
    'datafunction': 'quartic_data',
    'memory_depth': 2,

    # Timing (defaults: 10 ms heater, 50 kHz laser, 10 µs detector)
    't_phase_ms': 10.0,
    'f_laser_khz': 50.0,
    'det_window_us': 10.0,
    'max_swipe': 21,

    # Continuous swipe
    'use_continuous': False,
    'n_swipe': None,
    'swipe_span': np.pi / 20,

    # Training
    'lr': 0.03,
    'epochs': 50,
    'phase_idx': (0, 1),
    'n_photons': (1, 1),
    
    # Model initialization
    'init_theta': None,  # Will be set in trainer

    # Plotting
    'do_plot': True,

    # Sampler
    'n_samples': 1000
}
# =========================================================


def _resolve_n_swipe() -> int:
    """
    Resolves the number of swipes to use, either from config or by computing it.
    Returns:
        int: Number of swipes.
    """
    if config['n_swipe'] is not None:
        return config['n_swipe']
    auto = compute_n_swipe(
        config['t_phase_ms'],
        config['f_laser_khz'],
        config['det_window_us'],
        config['max_swipe'],
    )
    print(f"[timing] computed n_swipe = {auto}")
    return auto


def _run_training(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    cont: bool,
    n_samples: int
) -> None:
    """
    Runs the training process and plots results for both discrete and continuous modes.
    Args:
        X_train (np.ndarray): Training input data.
        y_train (np.ndarray): Training output data.
        X_test (np.ndarray): Test input data.
        y_test (np.ndarray): Test output data.
        cont (bool): Whether to use continuous-swipe training.
        n_samples (int): Number of samples for the Sampler.
    Returns:
        None
    """
    if cont:
        n_swipe = _resolve_n_swipe()
        config['n_swipe'] = n_swipe  # freeze for the rest of the run
        theta_opt, history = train_pytorch(
            X_train, y_train,
            memory_depth=config['memory_depth'],
            lr=config['lr'],
            epochs=config['epochs'],
            n_samples=n_samples,
            n_swipe=config['n_swipe'],
            swipe_span=config['swipe_span'],
            n_modes=3,
            encoding_mode=0,
            target_mode=(2,)
        )
    else:
        theta_opt, history = train_pytorch(
            X_train, y_train,
            memory_depth=config['memory_depth'],
            lr=config['lr'],
            epochs=config['epochs'],
            n_samples=n_samples,
            n_swipe=0,
            swipe_span=0.0,
            n_modes=3,
            encoding_mode=0,
            target_mode=(2,)
        )

    print("Optimized θ:", theta_opt)

    # ── predictions on dense grid ──
    if cont:
        preds = run_simulation_sequence_np(
            theta_opt, config['memory_depth'], n_samples,
            encoded_phases=2 * np.arccos(X_test),
            n_swipe=config['n_swipe'],
            swipe_span=config['swipe_span'],
            n_modes=3,
            encoding_mode=0,
            target_mode=(2,)
        )
    else:
        enc_test = 2 * np.arccos(X_test)
        preds = run_simulation_sequence_np(
            theta_opt, config['memory_depth'], n_samples,
            encoded_phases=enc_test,
            n_swipe=0,
            swipe_span=0.0,
            n_modes=3,
            encoding_mode=0,
            target_mode=(2,)
        )

    if config['do_plot']:
        # ————————————————————————————————————————————————————————————————
        # Plot 1: loss curve
        plt.figure(figsize=(9, 4))
        plt.plot(history)
        plt.yscale('log')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid()

        # Plot 2: data cloud + model fit
        plt.figure(figsize=(9, 5))
        plt.scatter(X_train, y_train, s=20, label='Original data', zorder=3)

        if cont:
            enc_swipe, _ = get_cont_swipe_data(X_train, y_train, n_swipe=config['n_swipe'], swipe_span=config['swipe_span'])
            X_swipe = np.cos(enc_swipe / 2)
            plt.scatter(X_swipe, np.repeat(y_train, config['n_swipe']), s=8, alpha=0.35, label=f'Swipe (n={config['n_swipe']})', zorder=2)

        plt.plot(X_test, y_test, label='Quartic', ls='--', zorder=1)
        plt.plot(X_test, preds, label='Model', c='red', zorder=4)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.grid()
        plt.legend()
        plt.tight_layout()

        # Plot 3: index vs encoded phase
        plt.figure(figsize=(9, 4))
        idx = np.arange(len(X_train))
        enc_orig = 2 * np.arccos(X_train)
        plt.plot(idx, enc_orig, '-o', label='Original enc φ', lw=1.5)

        if cont:
            # enc_swipe already computed above
            idx_swipe = np.repeat(idx, config['n_swipe'])
            plt.scatter(idx_swipe, enc_swipe, s=6, alpha=0.35, label='Swipe enc φ')

        plt.xlabel('Data index')
        plt.ylabel('Encoding phase [rad]')
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.show()


def main(n_samples: int, measured_data: Optional[str] = None, use_continuous: bool = False) -> None:
    """
    Main entry point for running the training and evaluation pipeline.
    Args:
        n_samples (int): Number of samples for the Sampler.
        measured_data (str | None): Path to measured data pickle file, or None for synthetic data.
        use_continuous (bool): Whether to use continuous-swipe training.
    Returns:
        None
    """
    if not isinstance(n_samples, int) or n_samples <= 0:
        raise ValueError(f"n_samples must be a positive int, got {n_samples!r}")
    if measured_data is not None:
        X_train, y_train = load_measurement_pickle(measured_data)
        # Synthetic test set: densify full phase range
        X_test = np.linspace(0.0, 1.0, 500)
        y_test = quartic_data(X_test)  # Placeholder when ground truth unknown
    else:
        X_train, y_train, X_test, y_test = get_data(
            config['n_data'],
            config['sigma_noise'],
            config['datafunction']
        )
    _run_training(X_train, y_train, X_test, y_test, cont=use_continuous, n_samples=n_samples)
    sim_logger.report()


def run_cli():
    """Command-line interface for the UQ-QNN training pipeline."""
    parser = argparse.ArgumentParser(description="UQ-QNN Photonic Training Pipeline")
    parser.add_argument("--n-samples", type=int, default=1000, 
                       help="Number of samples for the Sampler")
    parser.add_argument("--measured-data", type=str, default=None,
                       help="Path to measured data pickle file")
    parser.add_argument("--continuous", action="store_true",
                       help="Use continuous-swipe training")
    parser.add_argument("--epochs", type=int, default=50,
                       help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.03,
                       help="Learning rate")
    parser.add_argument("--no-plot", action="store_true",
                       help="Disable plotting")
    
    args = parser.parse_args()
    
    # Update config with CLI arguments
    config['epochs'] = args.epochs
    config['lr'] = args.lr
    config['do_plot'] = not args.no_plot
    
    main(args.n_samples, args.measured_data, args.continuous)


if __name__ == "__main__":
    run_cli()