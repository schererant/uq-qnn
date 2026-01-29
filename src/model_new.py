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
from src.quantum import MemristorCircuit, MemristorMegaCircuit

def MSEloss(y_train, prob_state_001):
    """ Compute MSE loss objective 001 state output.
    """
    loss = tf.square(tf.abs(y_train - prob_state_001))

    return loss

def NLLloss(y_train, prob_state_001, prob_state_010):
    """ Compute NLL loss objective 001 state output as mean and 010 state output as variance.
    """
    eps = 0.0001
    loss = tf.fraction(1, tf.pow(prob_state_010,2)+eps)*tf.square(tf.abs(y_train - prob_state_001))+ tf.log(tf.pow(prob_state_010,2)+eps)

    return loss

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

    num_phases = 2  # Change this number as needed for an arbitrary number of phases
    phases = [
        tf.Variable(rd.uniform(0.01, 1) * 2 * np.pi, dtype=tf.float32,
                    constraint=lambda z: tf.clip_by_value(z, 0, 2 * np.pi))
        for _ in range(num_phases)
    ]
    memristor_weight = tf.Variable(rd.uniform(0.01, 1), dtype=tf.float32,
                      constraint=lambda z: tf.clip_by_value(z, 0.01, 1))  # Memristor parameter

    
    logger.log_initial_training_phase(phases=phases, weights=[memristor_weight])
    
        
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
                    memristor_circuit = MemristorCircuit(phases[0], memristor_phase, phases[1], encoded_phases[i])
                    results = eng.run(
                        memristor_circuit.build_circuit())
                else:
                    memristor_phase = tf.acos(tf.sqrt(
                        tf.reduce_sum(memory_p1) / memory_depth +
                        memristor_weight * tf.reduce_sum(memory_p2) / memory_depth
                    ))
                    memristor_circuit = MemristorCircuit(phases[0], memristor_phase, phases[1], encoded_phases[i])
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
                loss += MSEloss(y_train[i],prob_state_001)

            # Compute gradients and update variables
            gradients = tape.gradient(loss, phases + [memristor_weight])
            optimizer.apply_gradients(zip(gradients, phases + [memristor_weight]))

            # Update progress bar with current loss
            pbar.set_postfix({'loss': f'{float(loss):.4f}'})
                    
            logger.log_training_step(step, loss, phases[0], phases[1], memristor_weight)


            res_mem[('loss', 'tr', step)] = [loss.numpy(), phases[0].numpy(), phases[1].numpy(), memristor_weight.numpy()]


    final_metrics = {
        'final_loss': float(loss),
        'final_phase1': float(phases[0]),
        'final_phase3': float(phases[1]),
        'final_memristor_weight': float(memristor_weight),
        'memory_depth': memory_depth,
        'training_steps': training_steps,
        'learning_rate': learning_rate,
        'cutoff_dim': cutoff_dim
    }
    logger.log_final_results(final_metrics)

    # Save trained model parameters
    trained_params = {
        'phase1': phases[0].numpy(),
        'phase3': phases[1].numpy(),
        'memristor_weight': memristor_weight.numpy(),
        'final_loss': loss.numpy(),
        'memory_depth': memory_depth,
        'training_steps': training_steps,
        'res_mem': res_mem
    }
    logger.save_model_artifact(trained_params, 'trained_parameters.pkl')

    # Plot training results
    plot_training_results(res_mem, f"{logger.base_dir}/plots/training_results_+{param_id}.png")
    
        
    return res_mem, phases[0], phases[1], memristor_weight