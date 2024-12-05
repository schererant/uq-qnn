import numpy as np
import tensorflow as tf
import strawberryfields as sf
from strawberryfields.ops import *
import random as rd
from datetime import datetime
from utils import log_training_loss
from tqdm import tqdm, trange
from plotting import plot_training_results, plot_predictions_new

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
rd.seed(42)


def build_circuit(phase1, memristor_weight, phase3, encoded_phases):
    """
    Constructs the quantum circuit with the given parameters.
    """
    circuit = sf.Program(3)
    with circuit.context as q:
        Vac     | q[0]
        Fock(1) | q[1]
        Vac     | q[2]
        
        # Input encoding MZI
        BSgate(np.pi/4, np.pi/2) | (q[0], q[1])
        Rgate(encoded_phases)           | q[1]
        BSgate(np.pi/4, np.pi/2) | (q[0], q[1])
        
        # First MZI
        BSgate(np.pi/4, np.pi/2) | (q[0], q[1])
        Rgate(phase1)             | q[1]
        BSgate(np.pi/4, np.pi/2) | (q[0], q[1])
        
        # Memristor (Second MZI)
        BSgate(np.pi/4, np.pi/2) | (q[1], q[2])
        Rgate(memristor_weight)             | q[1]
        BSgate(np.pi/4, np.pi/2) | (q[1], q[2])
        
        # Third MZI
        BSgate(np.pi/4, np.pi/2) | (q[0], q[1])
        Rgate(phase3)             | q[1]
        BSgate(np.pi/4, np.pi/2) | (q[0], q[1])
    return circuit


def train_memristor(X_train, 
                    y_train, 
                    memory_depth, 
                    training_steps, 
                    learning_rate, 
                    cutoff_dim, 
                    log_filepath: str = None,
                    log_path: str = None,
                    param_id: str = None, 
                    log = True, 
                    plot = True,
                    pickle = False):
    """
    Trains the memristor model using the training data.

    Args:
        x_train: Training input data.
        y_train: Training target data.
        memory_depth: Memory depth of the memristor.

    Returns:
        res_mem: Dictionary containing the training loss and parameters over iterations.
        phase1: Trained phase parameter 1.
        phase3: Trained phase parameter 3.
        memristor_weight: Trained weight parameter for the memristor update function.
    """
    # Create log file with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    # log_filepath = f"memristor_training_{timestamp}.txt"

    if log:
        with open(log_filepath, "a") as f:
                f.write("-" * 40 + "\n")
                f.write(f"qnn_hp_s{training_steps}_lr{learning_rate}_md{memory_depth}_cd{cutoff_dim}\n")
                f.write("-" * 40 + "\n\n")
                f.write("Hyperparameters:\n")
                f.write(" " * 2 + f"Memory Depth: {memory_depth}\n")
                f.write(" " * 2 + f"Training Steps: {training_steps}\n")
                f.write(" " * 2 + f"Learning Rate: {learning_rate}\n")
                f.write(" " * 2 + f"Cutoff Dimension: {cutoff_dim}\n")

                f.write("\n")
                f.write("-" * 20 + "\n")
                f.write("Training Log\n")
                f.write("-" * 20 + "\n")


    # Initialize variables and optimizer
    np.random.seed(42)
    tf.random.set_seed(42)
    rd.seed(42)

    phase1 = tf.Variable(rd.uniform(0.01, 1) * 2 * np.pi, dtype=tf.float32,
                       constraint=lambda z: tf.clip_by_value(z, 0, 2 * np.pi))
    phase3 = tf.Variable(rd.uniform(0.01, 1) * 2 * np.pi, dtype=tf.float32,
                       constraint=lambda z: tf.clip_by_value(z, 0, 2 * np.pi))
    memristor_weight = tf.Variable(rd.uniform(0.01, 1), dtype=tf.float32,
                      constraint=lambda z: tf.clip_by_value(z, 0.01, 1))  # Memristor parameter
    
    # Write header
    if log:
        with open(log_filepath, 'a') as f:
            f.write("\nInitial Parameters:\n")
            f.write(" " * 2 + f"Phase1: {float(phase1):.4f}\n")
            f.write(" " * 2 + f"Phase3: {float(phase3):.4f}\n")
            f.write(" " * 2 + f"Memristor Weight: {float(memristor_weight):.4f}\n")
            f.write("\nTraining Progress:\n")

        
        
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    eng = sf.Engine(backend="tf", backend_options={"cutoff_dim": cutoff_dim})
    res_mem = {}

    encoded_phases = tf.constant(2 * np.arccos(X_train), dtype=tf.float64)
    num_samples = len(encoded_phases)

    # Initialize memory variables
    memory_p1 = tf.Variable(np.zeros(memory_depth), dtype=tf.float32)
    memory_p2 = tf.Variable(np.zeros(memory_depth), dtype=tf.float32)
    cycle_index = 0

    # Training loop
    pbar = trange(training_steps, desc='Training', unit='step')
    for step in pbar:
        # Reset the engine if it has already been executed
        if eng.run_progs:
            eng.reset()

        with tf.GradientTape() as tape:
            loss = 0.0

            for i in tqdm(range(num_samples), 
                         desc=f'Step {step+1}/{training_steps}',
                         leave=False,
                         unit='sample'):
                time_step = i - cycle_index * memory_depth
                if time_step == memory_depth - 1:
                    cycle_index += 1

                if i == 0:
                    memristor_phase = tf.acos(tf.sqrt(0.5))
                    circuit = build_circuit(phase1, memristor_phase, phase3, encoded_phases[i])
                    results = eng.run(circuit)
                else:
                    memristor_phase = tf.acos(tf.sqrt(
                        tf.reduce_sum(memory_p1) / memory_depth +
                        memristor_weight * tf.reduce_sum(memory_p2) / memory_depth
                    ))
                    circuit = build_circuit(phase1, memristor_phase, phase3, encoded_phases[i])
                    results = eng.run(circuit)

                # Get probabilities from the circuit results
                prob = results.state.all_fock_probs()
                prob_state_001 = tf.cast(tf.math.real(prob[0, 0, 1]), dtype=tf.float64)
                prob_state_010 = tf.cast(tf.math.real(prob[0, 1, 0]), dtype=tf.float64)

                # Update memory variables
                memory_p1 = tf.tensor_scatter_nd_update(memory_p1, [[time_step % memory_depth]], [prob_state_010])
                memory_p2 = tf.tensor_scatter_nd_update(memory_p2, [[time_step % memory_depth]], [prob_state_001])

                # Compute the loss
                loss += tf.square(tf.abs(y_train[i] - prob_state_001))

            # Compute gradients and update variables
            gradients = tape.gradient(loss, [phase1, phase3, memristor_weight])
            optimizer.apply_gradients(zip(gradients, [phase1, phase3, memristor_weight]))

            # Update progress bar with current loss
            pbar.set_postfix({'loss': f'{float(loss):.4f}'})

            # Log results every step
            # log_training_loss(log_filepath, step, loss, phase1, phase3, memristor_weight)
            if log:
                with open(log_filepath, 'a') as f:
                    f.write(" " * 2 + f"Step {step:4d}: Loss = {loss:.4f}, "
                        f"Phase1 = {float(phase1):.4f}, "
                        f"Phase3 = {float(phase3):.4f}, "
                        f"Weight = {float(memristor_weight):.4f}\n")


            res_mem[('loss', 'tr', step)] = [loss.numpy(), phase1.numpy(), phase3.numpy(), memristor_weight.numpy()]
            # print(f"Loss at step {step + 1}: {loss.numpy()}")

    # print(f"Final loss: {loss.numpy()}")
    # print(f"Optimal parameters: phase1={phase1.numpy()}, phase3={phase3.numpy()}, memristor_weight={memristor_weight.numpy()}")

    if log:
        with open(log_filepath, 'a') as f:
            f.write("\nFinal Parameters:\n") 
            f.write(" " * 2 + f"Phase1: {float(phase1):.4f}\n")
            f.write(" " * 2 + f"Phase3: {float(phase3):.4f}\n")
            f.write(" " * 2 + f"Memristor Weight: {float(memristor_weight):.4f}\n")
            f.write("\nTraining Summary:\n")
            f.write(" " * 2 + f"Initial Loss: {res_mem[('loss', 'tr', 0)][0]:.4f}\n")
            f.write(" " * 2 + f"Final Loss: {loss:.4f}\n")

    if plot:
        plot_training_results(res_mem, log_path+f"training_results_{param_id}.png")
    
    if pickle:
        # Prepare data to be saved
        trained_params = {
            'res_mem': res_mem,
            'phase1': phase1.numpy(),
            'phase3': phase3.numpy(),
            'memristor_weight': memristor_weight.numpy()
        }

        # Define the filename
        pickle_filename = f"{log_path}trained_params_{param_id}.pkl"

        # Save the data to a pickle file
        with open(pickle_filename, 'wb') as f:
            pickle.dump(trained_params, f)
            
    return res_mem, phase1, phase3, memristor_weight

def predict_memristor(X_test: np.ndarray, 
                      y_test: np.ndarray, 
                      memory_depth: int, 
                      phase1: float, 
                      phase3: float, 
                      memristor_weight: float, 
                      stochastic: bool, 
                      samples: int, 
                      var: float, 
                      cutoff_dim: int, 
                      log_filepath: str = None,
                      log_path: str = None,
                      param_id: str = None,
                      log = True,
                      plot = True):
    """
    Uses the trained memristor model to make predictions on test data.
    """

    if log:
        with open(log_filepath, "a") as f:
                f.write("\n")
                f.write("-" * 20 + "\n")
                f.write("Prediction Log\n")
                f.write("-" * 20 + "\n")

    # Write header and initial parameters
    if log:
        with open(log_filepath, 'a') as f:
            f.write("\nPrediction Parameters:\n")
            f.write(" " * 2 + f"Memory Depth: {memory_depth}\n")
            f.write(" " * 2 + f"Phase1: {float(phase1):.4f}\n")
            f.write(" " * 2 + f"Phase3: {float(phase3):.4f}\n")
            f.write(" " * 2 + f"Memristor Weight: {float(memristor_weight):.4f}\n")
            f.write(" " * 2 + f"Stochastic: {stochastic}\n")
            if stochastic:
                f.write(" " * 2 + f"Number of Samples: {samples}\n")
                f.write(" " * 2 + f"Variance: {var}\n")
            

    eng = sf.Engine(backend="tf", backend_options={"cutoff_dim": cutoff_dim})
    encoded_phases = tf.constant(2 * np.arccos(X_test), dtype=tf.float64)

    # Initialize lists to store predictions and targets
    all_predictions = []
    targets = []

    # Initialize memory variables
    memory_p1 = tf.Variable(np.zeros(memory_depth), dtype=tf.float32)
    memory_p2 = tf.Variable(np.zeros(memory_depth), dtype=tf.float32)
    cycle_index = 0

    # print("Predicting on test data...")
    sample_pbar = trange(samples, desc='Prediction Samples', unit='sample')
    for sample in sample_pbar:
        sample_predictions = []

        # Inner progress bar for phases
        phase_pbar = tqdm(range(len(encoded_phases)), 
                         desc=f'Sample {sample+1}/{samples}',
                         leave=False,
                         unit='phase')

        for i in phase_pbar:
            time_step = i - cycle_index * memory_depth
            if time_step == memory_depth - 1:
                cycle_index += 1
            if stochastic:
                phase1_sample = np.random.normal(phase1.numpy(), var)
                phase3_sample = np.random.normal(phase3.numpy(), var)
            else:
                phase1_sample = phase1.numpy()
                phase3_sample = phase3.numpy()
            if i == 0:
                memristor_phase = tf.acos(tf.sqrt(0.5))
            else:
                memristor_phase = tf.acos(tf.sqrt(
                    tf.reduce_sum(memory_p1) / memory_depth +
                    memristor_weight * tf.reduce_sum(memory_p2) / memory_depth
                ))

            circuit = build_circuit(phase1_sample, memristor_phase, phase3_sample, encoded_phases[i])
            results = eng.run(circuit)

            # Get probabilities from the circuit results
            prob = results.state.all_fock_probs()
            prob_state_001 = tf.cast(tf.math.real(prob[0, 0, 1]), dtype=tf.float32)
            prob_state_010 = tf.cast(tf.math.real(prob[0, 1, 0]), dtype=tf.float32)

            # Update memory variables
            memory_p1 = tf.tensor_scatter_nd_update(memory_p1, [[time_step % memory_depth]], [prob_state_010])
            memory_p2 = tf.tensor_scatter_nd_update(memory_p2, [[time_step % memory_depth]], [prob_state_001])

            sample_predictions.append(prob_state_001.numpy())
            if sample == 0:
                targets.append(float(y_test[i].numpy()))

            # Update inner progress bar with current probabilities
            phase_pbar.set_postfix({
                'prob_001': f'{float(prob_state_001):.4f}',
                'prob_010': f'{float(prob_state_010):.4f}'
            })

        all_predictions.append(sample_predictions)

    # Convert all_predictions to a NumPy array for easier manipulation
    all_predictions = np.array(all_predictions)

    if stochastic:
        # Calculate mean and standard deviation along the column axis
        final_predictions = np.mean(all_predictions, axis=0)
        predictive_uncertainty = np.std(all_predictions, axis=0)
    else:
        final_predictions = all_predictions[0]
        predictive_uncertainty = np.array([])
        targets = np.array(targets)

    # Log final results
    if log:
        with open(log_filepath, 'a') as f:
            f.write("\nPrediction Summary:\n")
            f.write(" " * 2 + f"Number of test samples: {len(X_test)}\n")
            f.write(" " * 2 + f"Mean prediction: {np.mean(final_predictions):.4f}\n")
            if stochastic:
                f.write(" " * 2 + f"Mean predictive uncertainty: {np.mean(predictive_uncertainty):.4f}\n")
            f.write(" " * 2 + f"Mean absolute error: {np.mean(np.abs(final_predictions - targets)):.4f}\n")

    if plot:
        plot_predictions_new(X_test, y_test, final_predictions, predictive_uncertainty, log_path+f"prediction_results_{param_id}.png")

    return final_predictions, targets, predictive_uncertainty

