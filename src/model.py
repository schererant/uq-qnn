import numpy as np
import tensorflow as tf
import strawberryfields as sf
from strawberryfields.ops import *
import random as rd
from datetime import datetime
from tqdm import tqdm, trange
import pickle


from src.plotting import plot_training_results, plot_predictions_new
from src.utils import log_training_loss
from src.logger import ExperimentLogger
from src.quantum import MemristorCircuit


def train_memristor(X_train, 
                    y_train, 
                    memory_depth, 
                    training_steps, 
                    learning_rate, 
                    cutoff_dim, 
                    logger: ExperimentLogger,
                    log_filepath: str = None,
                    log_path: str = None,
                    param_id: str = None, 
                    plot = True,
                    plot_path: str = None):
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

    
    logger.log_initial_training_phase(phase1, phase3, memristor_weight)
    
        
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

                    # Init Memristor Circuit
                    memristor_circuit = MemristorCircuit(phase1, memristor_phase, phase3, encoded_phases[i])
                    results = eng.run(
                        memristor_circuit.build_circuit())
                else:
                    memristor_phase = tf.acos(tf.sqrt(
                        tf.reduce_sum(memory_p1) / memory_depth +
                        memristor_weight * tf.reduce_sum(memory_p2) / memory_depth
                    ))
                    memristor_circuit = MemristorCircuit(phase1, memristor_phase, phase3, encoded_phases[i])
                    results = eng.run(
                        memristor_circuit.build_circuit())

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
                    
            logger.log_training_step(step, loss, phase1, phase3, memristor_weight)


            res_mem[('loss', 'tr', step)] = [loss.numpy(), phase1.numpy(), phase3.numpy(), memristor_weight.numpy()]


    final_metrics = {
        'final_loss': float(loss),
        'final_phase1': float(phase1),
        'final_phase3': float(phase3),
        'final_memristor_weight': float(memristor_weight),
        'memory_depth': memory_depth,
        'training_steps': training_steps,
        'learning_rate': learning_rate,
        'cutoff_dim': cutoff_dim
    }
    logger.log_final_results(final_metrics)

    # Save trained model parameters
    trained_params = {
        'phase1': phase1.numpy(),
        'phase3': phase3.numpy(),
        'memristor_weight': memristor_weight.numpy(),
        'final_loss': loss.numpy(),
        'memory_depth': memory_depth,
        'training_steps': training_steps,
        'res_mem': res_mem
    }
    logger.save_model_artifact(trained_params, 'trained_parameters.pkl')

    # Plot training results
    plot_training_results(res_mem, f"{logger.base_dir}/plots/training_results_+{param_id}.png")
    
        
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
                      logger: ExperimentLogger,
                      param_id: str = None):
    """
    Uses the trained memristor model to make predictions on test data.
    """

    eng = sf.Engine(backend="tf", backend_options={"cutoff_dim": cutoff_dim})
    encoded_phases = tf.constant(2 * np.arccos(X_test), dtype=tf.float64)

    # Initialize lists to store predictions and targets
    all_predictions = []
    targets = []

    # Initialize memory variables
    memory_p1 = tf.Variable(np.zeros(memory_depth), dtype=tf.float32)
    memory_p2 = tf.Variable(np.zeros(memory_depth), dtype=tf.float32)
    cycle_index = 0

    if stochastic:
        print(f"Running {samples} samples with variance {var}...")
    else:
        print("Running deterministic prediction...")
        samples = 1

    # print("Predicting on test data...")
    sample_pbar = trange(samples, desc='Prediction Samples', unit='sample')
    for sample in sample_pbar:
        sample_predictions = []

        if stochastic:
            phase1_sample = np.random.normal(phase1, var)
            # memristor_sample = np.random.normal(memristor_weight, var)
            phase3_sample = np.random.normal(phase3, var)
        else:
            phase1_sample = phase1
            # memristor_sample = memristor_weight
            phase3_sample = phase3


        # Inner progress bar for phases
        phase_pbar = tqdm(range(len(encoded_phases)), 
                         desc=f'Sample {sample+1}/{samples}',
                         leave=False,
                         unit='phase')

        for i in phase_pbar:
            time_step = i - cycle_index * memory_depth
            if time_step == memory_depth - 1:
                cycle_index += 1
            if i == 0:
                memristor_phase = tf.acos(tf.sqrt(0.5))
            else:
                memristor_phase = tf.acos(tf.sqrt(
                    tf.reduce_sum(memory_p1) / memory_depth +
                    memristor_weight * tf.reduce_sum(memory_p2) / memory_depth
                ))


            memristor_circuit = MemristorCircuit(phase1_sample, memristor_phase, phase3_sample, encoded_phases[i])
            results = eng.run(memristor_circuit.build_circuit())

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
            # Compute the loss
            prob_state_001 = tf.cast(prob_state_001, dtype=tf.float64)
            loss = tf.square(tf.abs(y_test[i] - prob_state_001))
            logger.log_prediction_step(i, loss, phase1_sample, phase3_sample, memristor_weight)

        all_predictions.append(sample_predictions)

    # Convert all_predictions to a NumPy array for easier manipulation
    all_predictions = np.array(all_predictions)

    if stochastic:
        # Calculate mean and standard deviation along the column axis
        final_predictions = np.mean(all_predictions, axis=0)
        predictive_uncertainty = np.std(all_predictions, axis=0)
        logger.log_prediction(final_predictions, predictive_uncertainty, samples)
        plot_predictions_new(X_test, y_test, final_predictions, predictive_uncertainty, f"{logger.base_dir}/plots/prediction_results_sample{samples}_{param_id}.png")
    else:
        final_predictions = all_predictions[0]
        predictive_uncertainty = np.array([])
        targets = np.array(targets)
        logger.log_prediction(final_predictions)
        plot_predictions_new(X_test, y_test, final_predictions, predictive_uncertainty, f"{logger.base_dir}/plots/prediction_results_deterministic_{param_id}.png")

    


    return final_predictions, targets, predictive_uncertainty, all_predictions

