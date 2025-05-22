import argparse
from config.default import default_config
from data import get_data_source
from circuits import get_circuit
from simulation import SimulationRunner
from autograd import get_gradient_method
from loss import get_loss_function
from training import Trainer
from utils.helpers import plot_training_curve, plot_loss_ascii, plot_perceval_style, set_seed, load_model_params
from loss.mse import MSELoss
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Quantum Neural Network Trainer")
    parser.add_argument('--datafunction', type=str, default=default_config['datafunction'], help='Data function (quartic, custom, measured)')
    parser.add_argument('--n_data', type=int, default=default_config['n_data'], help='Number of training data points')
    parser.add_argument('--sigma_noise', type=float, default=default_config['sigma_noise'], help='Noise std for synthetic data')
    parser.add_argument('--circuit', type=str, default='full', help='Circuit type (encoding, memristor, full)')
    parser.add_argument('--grad', type=str, default='psr', help='Gradient method (psr, finite_diff)')
    parser.add_argument('--loss', type=str, default='mse', help='Loss function (mse, custom)')
    parser.add_argument('--epochs', type=int, default=default_config['epochs'], help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=default_config['lr'], help='Learning rate')
    parser.add_argument('--n_samples', type=int, default=default_config['n_samples'], help='Number of samples for simulation')
    parser.add_argument('--plotting', type=str, choices=['none', 'terminal', 'pyplot'], help='Plotting mode: none, terminal, pyplot')
    parser.add_argument('--inference', action='store_true', help='Run inference only (no training)')
    parser.add_argument('--param_path', type=str, default=None, help='Path to model parameters for inference (overrides config)')
    parser.add_argument('--params', type=float, nargs='+', default=None, help='Hand-given model parameters for inference (overrides file)')
    parser.add_argument('--use-torch', action='store_true', help='Use PyTorch-based training (PSR autograd)')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    config = default_config.copy()
    config.update(vars(args))
    set_seed(config['seed'])
    # 1. Data source
    data_source = get_data_source(
        name=config['datafunction'],
        n_data=config['n_data'],
        sigma_noise=config['sigma_noise'],
        seed=config['seed']
    )
    # 2. Circuit
    circuit = get_circuit(config['circuit'])
    # 3. Simulation runner
    simulation_runner = SimulationRunner(circuit, memory_depth=config['memory_depth'])
    # 4. Gradient method
    grad_method = get_gradient_method(
        config['grad'],
        simulation_runner=simulation_runner,
        memory_depth=config['memory_depth'],
        phase_idx=config['phase_idx'],
        n_photons=config['n_photons'],
        n_samples=config['n_samples'],
        n_swipe=config.get('n_swipe', 0) or 0,
        swipe_span=config.get('swipe_span', 0.0) or 0.0
    )
    # 5. Loss function
    loss_fn = get_loss_function(config['loss'])
    # 6. Trainer
    trainer = Trainer(
        data_source=data_source,
        circuit=circuit,
        simulation_runner=simulation_runner,
        grad_method=grad_method,
        loss_fn=loss_fn,
        config=config
    )
    # Inference mode
    if args.inference:
        # Use hand-given params if provided
        if args.params is not None:
            params = np.array(args.params)
        else:
            param_path = args.param_path or config.get('param_save_path', 'model_params.npy')
            params = load_model_params(param_path)
        X_train, y_train, X_test, y_test = data_source.get_data()
        enc_test = data_source.encode_phase(X_test)
        preds = simulation_runner.run_sequence(params, enc_test, config['n_samples'])
        # Print loss if ground truth is available
        mse_loss = MSELoss()
        if y_test is not None:
            mse = mse_loss.compute(preds, y_test)
            print(f"Inference MSE loss (test set): {mse:.6f}")
        # Also compute and print training set loss with same params
        preds_train = simulation_runner.run_sequence(params, data_source.encode_phase(X_train), config['n_samples'])
        mse_train = mse_loss.compute(preds_train, y_train)
        print(f"Inference MSE loss (training set): {mse_train:.6f}")
        if config['plotting'] == 'terminal':
            print('Inference complete.')
        elif config['plotting'] == 'pyplot':
            plot_perceval_style(X_train, y_train, X_test, y_test, preds, history=None, config=config)
        else:
            print('Inference complete.')
        exit(0)
    # 7. Run training
    if config.get('use_torch', False):
        from training import train_pytorch
        X_train, y_train, X_test, y_test = data_source.get_data()
        final_params, loss_history = train_pytorch(
            circuit,
            X_train, y_train,
            memory_depth=config['memory_depth'],
            lr=config['lr'],
            epochs=config['epochs'],
            phase_idx=config['phase_idx'],
            n_photons=config['n_photons'],
            seed=config['seed'],
            n_samples=config['n_samples'],
            n_swipe=config.get('n_swipe', 0) or 0,
            swipe_span=config.get('swipe_span', 0.0) or 0.0
        )
        from utils.helpers import save_model_params
        save_model_params(final_params, config.get('param_save_path', 'model_params.npy'))
    else:
        loss_history, final_params = trainer.run()
    # 8. Plotting
    if config['plotting'] == 'terminal':
        plot_loss_ascii(loss_history)
    elif config['plotting'] == 'pyplot':
        # Gather all data for Perceval-style plot
        X_train, y_train, X_test, y_test = data_source.get_data()
        enc_train = data_source.encode_phase(X_train)
        # Get predictions on test set
        preds = simulation_runner.run_sequence(final_params, data_source.encode_phase(X_test), config['n_samples'])
        # Optionally, handle swipe data if needed
        plot_perceval_style(X_train, y_train, X_test, y_test, preds, loss_history, config=config)
    # If 'none', do nothing 

    # 9. Print simulation statistics
    simulation_runner.logger.report()