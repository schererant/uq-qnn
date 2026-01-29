import perceval as pcvl
import numpy as np
# from scipy.optimize import minimize # Optional for more advanced optimizers
import random as rd
from tqdm import trange, tqdm
import pickle
import math # For pi

# Assuming these imports are still valid or adapted
from src.plotting import plot_training_results
from src.utils import log_training_loss # May need adaptation
from src.logger import ExperimentLogger # May need adaptation

# --- Helper Functions ---

def build_perceval_memristor_circuit(phase1, memristor_phase, phase3, encoded_phase):
    """Builds the 3-mode memristor circuit using Perceval components."""
    # Assumes SF Rgate(phi) corresponds to pcvl PS(phi). Verify if needed.
    c = pcvl.Circuit(3)
    # Input encoding MZI (modes 0, 1)
    c.add((0, 1), pcvl.BS())
    c.add((1,), pcvl.PS(phi=encoded_phase))
    c.add((0, 1), pcvl.BS())
    # First MZI (modes 0, 1)
    c.add((0, 1), pcvl.BS())
    c.add((1,), pcvl.PS(phi=phase1))
    c.add((0, 1), pcvl.BS())
    # Memristor MZI (modes 1, 2)
    c.add((1, 2), pcvl.BS())
    c.add((1,), pcvl.PS(phi=memristor_phase))
    c.add((1, 2), pcvl.BS())
    # Third MZI (modes 0, 1)
    c.add((0, 1), pcvl.BS())
    c.add((1,), pcvl.PS(phi=phase3))
    c.add((0, 1), pcvl.BS())
    return c

def run_simulation_sequence(params, encoded_phases_all, memory_depth):
    """Runs the simulation over the sequence, updating memory."""
    phase1, phase3, memristor_weight = params
    num_samples = len(encoded_phases_all)

    memory_p1 = np.zeros(memory_depth, dtype=np.float64)
    memory_p2 = np.zeros(memory_depth, dtype=np.float64)

    predictions_001 = np.zeros(num_samples, dtype=np.float64)
    # predictions_010 = np.zeros(num_samples, dtype=np.float64) # Only needed internally

    backend = "SLOS"  # Efficient backend for Fock states
    input_state = pcvl.BasicState([0, 1, 0])
    state_001 = pcvl.BasicState([0, 0, 1])
    state_010 = pcvl.BasicState([0, 1, 0])
    # Pre-create Analyzer instance for potential reuse if parameters don't change structure
    # Note: Analyzer needs specific states, so maybe create inside loop if states change?
    # Here output states are fixed.

    for i in range(num_samples):
        time_step = i % memory_depth

        if i == 0:
            memristor_phase = np.pi / 4.0 # acos(sqrt(0.5))
        else:
            # Calculate memristor phase based on memory
            mem_term1 = np.sum(memory_p1) / memory_depth
            mem_term2 = memristor_weight * np.sum(memory_p2) / memory_depth
            # Clip argument for numerical stability before sqrt and acos
            sqrt_arg = np.clip(mem_term1 + mem_term2, 1e-9, 1.0 - 1e-9)
            memristor_phase = np.arccos(np.sqrt(sqrt_arg))

        circuit = build_perceval_memristor_circuit(phase1, memristor_phase, phase3, encoded_phases_all[i])
        proc = pcvl.Processor(backend, circuit)
        proc.with_input(input_state)
        # Use Analyzer for exact probabilities
        analyzer = pcvl.algorithm.Analyzer(proc, input_states=[input_state], output_states=[state_001, state_010])
        probs = analyzer.probs()

        prob_state_001 = probs['results'].get(state_001, 0.0)
        prob_state_010 = probs['results'].get(state_010, 0.0)

        predictions_001[i] = prob_state_001
        # predictions_010[i] = prob_state_010 # Store if needed outside

        # Update memory
        memory_p1[time_step] = prob_state_010
        memory_p2[time_step] = prob_state_001

    # Return only the prediction needed for loss calculation
    return predictions_001

def mse_loss(y_true, y_pred):
    """Computes Mean Squared Error loss."""
    return np.mean(np.square(np.asarray(y_true) - np.asarray(y_pred)))

def objective_function(params, encoded_phases_all, y_train, memory_depth):
    """Calculates the loss for a given set of parameters."""
    predictions_001 = run_simulation_sequence(params, encoded_phases_all, memory_depth)
    return mse_loss(y_train, predictions_001)

def compute_gradients_combined(params, encoded_phases_all, y_train, memory_depth):
    """Computes gradients using parameter-shift for phases and finite diff for weight."""
    phase1, phase3, memristor_weight = params
    grads = np.zeros_like(params, dtype=np.float64)
    epsilon = 1e-7  # Step for finite difference

    # Parameter-shift for phase1
    params_p1_plus = np.array([phase1 + np.pi / 2, phase3, memristor_weight])
    params_p1_minus = np.array([phase1 - np.pi / 2, phase3, memristor_weight])
    loss_p1_plus = objective_function(params_p1_plus, encoded_phases_all, y_train, memory_depth)
    loss_p1_minus = objective_function(params_p1_minus, encoded_phases_all, y_train, memory_depth)
    grads[0] = 0.5 * (loss_p1_plus - loss_p1_minus)

    # Parameter-shift for phase3
    params_p3_plus = np.array([phase1, phase3 + np.pi / 2, memristor_weight])
    params_p3_minus = np.array([phase1, phase3 - np.pi / 2, memristor_weight])
    loss_p3_plus = objective_function(params_p3_plus, encoded_phases_all, y_train, memory_depth)
    loss_p3_minus = objective_function(params_p3_minus, encoded_phases_all, y_train, memory_depth)
    grads[1] = 0.5 * (loss_p3_plus - loss_p3_minus)

    # Finite difference for memristor_weight
    params_mw_plus = np.array([phase1, phase3, memristor_weight + epsilon])
    params_mw_minus = np.array([phase1, phase3, memristor_weight - epsilon])
    # Ensure weights stay within bounds for simulation runs during gradient calculation
    params_mw_plus[2] = np.clip(params_mw_plus[2], 0.01, 1.0)
    params_mw_minus[2] = np.clip(params_mw_minus[2], 0.01, 1.0)
    loss_mw_plus = objective_function(params_mw_plus, encoded_phases_all, y_train, memory_depth)
    loss_mw_minus = objective_function(params_mw_minus, encoded_phases_all, y_train, memory_depth)
    grads[2] = (loss_mw_plus - loss_mw_minus) / (2 * epsilon)

    return grads

# --- Main Training Function ---

def train_memristor(X_train,
                    y_train,
                    memory_depth,
                    training_steps,
                    learning_rate,
                    cutoff_dim, # Kept for signature consistency, not used by SLOS backend
                    logger: ExperimentLogger,
                    log_filepath: str = None, # Removed as unused in original call logic
                    log_path: str = None,     # Removed as unused
                    param_id: str = None,
                    plot = True,
                    plot_path: str = None): # plot_path used directly now
    """
    Trains the memristor model using Perceval and NumPy.

    Args:
        X_train: Training input data (numpy array expected).
        y_train: Training target data (numpy array expected).
        memory_depth: Memory depth of the memristor.
        training_steps: Number of optimization steps.
        learning_rate: Learning rate for gradient descent.
        cutoff_dim: (Ignored for SLOS backend) Max Fock state number for CV simulation.
        logger: An instance of ExperimentLogger (needs adaptation).
        param_id: Identifier string for logging/plotting.
        plot: Boolean flag to enable plotting.
        plot_path: Base directory for saving plots.


    Returns:
        res_mem: Dictionary containing the training loss and parameters over iterations.
        phase1: Trained phase parameter 1.
        phase3: Trained phase parameter 3.
        memristor_weight: Trained weight parameter for the memristor update function.
    """
    # Ensure input data are numpy arrays
    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)

    # Initialize variables
    np.random.seed(42)
    rd.seed(42)

    phase1_init = rd.uniform(0.01, 1) * 2 * np.pi
    phase3_init = rd.uniform(0.01, 1) * 2 * np.pi
    memristor_weight_init = rd.uniform(0.01, 1)
    params = np.array([phase1_init, phase3_init, memristor_weight_init], dtype=np.float64)

    # Adapt logger call if necessary
    # logger.log_initial_training_phase(phases=[params[0], params[1]], weights=[params[2]])

    res_mem = {} # Dictionary to store training history

    # Pre-encode phases (assuming X_train needs arccos like in original)
    # Verify if X_train is already the phase or needs transformation
    encoded_phases_all = 2 * np.arccos(np.clip(X_train, -1.0 + 1e-9, 1.0 - 1e-9)) # Clip for domain safety

    # Training loop using basic gradient descent
    pbar = trange(training_steps, desc='Training (Perceval/NumPy)', unit='step')
    for step in pbar:
        # Compute gradients for the current parameters
        # This involves multiple simulations inside compute_gradients_combined
        gradients = compute_gradients_combined(params, encoded_phases_all, y_train, memory_depth)

        # Apply gradient descent step
        params -= learning_rate * gradients

        # Apply constraints (clipping)
        params[0] = np.clip(params[0], 0, 2 * np.pi)  # phase1
        params[1] = np.clip(params[1], 0, 2 * np.pi)  # phase3
        params[2] = np.clip(params[2], 0.01, 1.0)     # memristor_weight

        # Calculate current loss for logging/display (optional, adds overhead)
        # You could calculate loss less frequently if needed
        if step % 10 == 0 or step == training_steps - 1: # Example: Calculate loss every 10 steps
             current_loss = objective_function(params, encoded_phases_all, y_train, memory_depth)
             pbar.set_postfix({'loss': f'{current_loss:.4f}'})
             # Store results
             res_mem[('loss', 'tr', step)] = [current_loss, params[0], params[1], params[2]]
             # Adapt logger call
             # logger.log_training_step(step, current_loss, params[0], params[1], params[2])
        else:
             # Store results without recalculating loss (use previous loss for display if needed)
             if step > 0:
                 last_loss = res_mem[('loss', 'tr', max(k[2] for k in res_mem if k[0]=='loss'))][0]
                 pbar.set_postfix({'loss': f'{last_loss:.4f}'})
                 res_mem[('loss', 'tr', step)] = [last_loss, params[0], params[1], params[2]] # Store params anyway
             else:
                 pbar.set_postfix({'loss': 'calculating...'})
                 res_mem[('loss', 'tr', step)] = [np.nan, params[0], params[1], params[2]] # Store params anyway


    # Final loss after training
    final_loss = objective_function(params, encoded_phases_all, y_train, memory_depth)
    print(f"Final Loss: {final_loss}")

    # Adapt logger call
    final_metrics = {
        'final_loss': final_loss,
        'final_phase1': params[0],
        'final_phase3': params[1],
        'final_memristor_weight': params[2],
        'memory_depth': memory_depth,
        'training_steps': training_steps,
        'learning_rate': learning_rate,
        'cutoff_dim': cutoff_dim # Include for consistency
    }
    # logger.log_final_results(final_metrics)

    # Save trained model parameters (adapt logger call or use pickle directly)
    trained_params = {
        'phase1': params[0],
        'phase3': params[1],
        'memristor_weight': params[2],
        'final_loss': final_loss,
        'memory_depth': memory_depth,
        'training_steps': training_steps,
        'res_mem': res_mem # Store the history
    }
    # Example using pickle directly if logger is not adapted:
    # if logger and hasattr(logger, 'base_dir'):
    #    save_path = f"{logger.base_dir}/trained_parameters_perceval_{param_id}.pkl"
    #    with open(save_path, 'wb') as f:
    #        pickle.dump(trained_params, f)
    #    print(f"Saved parameters to {save_path}")


    # Plot training results (adapt plotting call)
    if plot and plot_path:
         # Ensure plot_path exists
         import os
         os.makedirs(plot_path, exist_ok=True)
         plot_save_path = os.path.join(plot_path, f"training_results_perceval_{param_id}.png")
         try:
             # Make sure plot_training_results accepts the res_mem format
             plot_training_results(res_mem, plot_save_path)
             print(f"Saved plot to {plot_save_path}")
         except Exception as e:
             print(f"Failed to plot training results: {e}")


    return res_mem, params[0], params[1], params[2]

# ... (Keep the predict_memristor function, but it will also need adaptation
#      to use run_simulation_sequence instead of the TF engine) ...

# --- Example Adaptation for predict_memristor (Conceptual) ---

def predict_memristor_perceval(X_test: np.ndarray,
                               y_test: np.ndarray,
                               memory_depth: int,
                               phase1: float,
                               phase3: float,
                               memristor_weight: float,
                               stochastic: bool,
                               samples: int,
                               var: float,
                               cutoff_dim: int, # Ignored
                               logger: ExperimentLogger, # Adapt usage
                               param_id: str = None):
    """
    Uses the trained memristor model (Perceval version) to make predictions.
    (This needs similar adaptation as the training function)
    """
    X_test = np.asarray(X_test, dtype=np.float64)
    y_test = np.asarray(y_test, dtype=np.float64) # Ensure numpy array

    encoded_phases_all = 2 * np.arccos(np.clip(X_test, -1.0 + 1e-9, 1.0 - 1e-9))

    all_predictions_np = []
    targets = y_test # Targets are directly from y_test

    if not stochastic:
        samples = 1
        print("Running deterministic prediction (Perceval)...")
    else:
        print(f"Running {samples} stochastic samples with variance {var} (Perceval)...")

    sample_pbar = trange(samples, desc='Prediction Samples (Perceval)', unit='sample')
    for sample in sample_pbar:
        if stochastic:
            # Sample parameters around the mean trained values
            phase1_sample = np.random.normal(phase1, np.sqrt(var)) # var is variance, stddev is sqrt(var)
            phase3_sample = np.random.normal(phase3, np.sqrt(var))
            # Weight sampling might need clipping if variance is large
            memristor_weight_sample = np.clip(np.random.normal(memristor_weight, np.sqrt(var)), 0.01, 1.0)
        else:
            phase1_sample = phase1
            phase3_sample = phase3
            memristor_weight_sample = memristor_weight

        current_params = np.array([phase1_sample, phase3_sample, memristor_weight_sample])

        # Run the simulation sequence for this sample
        sample_preds = run_simulation_sequence(current_params, encoded_phases_all, memory_depth)
        all_predictions_np.append(sample_preds)

        # Optional: Log prediction steps if needed (adapt logger)
        # loss_sample = mse_loss(targets, sample_preds)
        # logger.log_prediction_step(...) # Adapt logger call

    all_predictions_np = np.array(all_predictions_np) # Shape: (samples, num_test_points)

    if stochastic:
        final_predictions = np.mean(all_predictions_np, axis=0)
        predictive_uncertainty = np.std(all_predictions_np, axis=0)
        # logger.log_prediction(final_predictions, predictive_uncertainty, samples) # Adapt logger
        plot_save_path = f"{logger.base_dir}/plots/prediction_results_stochastic_{samples}_{param_id}.png"
        plot_predictions_new(X_test, targets, final_predictions, predictive_uncertainty, plot_save_path) # Adapt plot call if needed
    else:
        final_predictions = all_predictions_np[0]
        predictive_uncertainty = np.array([]) # No uncertainty for deterministic
        # logger.log_prediction(final_predictions) # Adapt logger
        plot_save_path = f"{logger.base_dir}/plots/prediction_results_deterministic_{param_id}.png"
        plot_predictions_new(X_test, targets, final_predictions, predictive_uncertainty, plot_save_path) # Adapt plot call if needed

    return final_predictions, targets, predictive_uncertainty, all_predictions_np


# Keep other functions like MSEloss, NLLloss etc. if they are still needed elsewhere
# Remove the TensorFlow/StrawberryFields based train/predict functions if replacing them