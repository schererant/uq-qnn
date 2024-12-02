from datetime import datetime
import tensorflow as tf
import numpy as np
import random as rd
import strawberryfields as sf
from ..quantum.memristor_chip import build_circuit


def predict_memristor(x_test, y_test, memory_depth, phase1, phase3, memristor_weight, stochastic: bool = True, samples: int = 1, var: float = 0.1, cutoff_dim = 4, filename = None):
    """
    Uses the trained memristor model to make predictions on test data.
    """
    # Log file handling
    log_filepath = filename

    with open(log_filepath, "a") as f:
            f.write("\n")
            f.write("-" * 20 + "\n")
            f.write("Prediction Log\n")
            f.write("-" * 20 + "\n")

    # Write header and initial parameters
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
        f.write("\nPrediction Progress:\n")

    eng = sf.Engine(backend="tf", backend_options={"cutoff_dim": cutoff_dim})
    encoded_phases = tf.constant(2 * np.arccos(x_test), dtype=tf.float64)

    # Initialize lists to store predictions and targets
    all_predictions = []
    targets = []

    # Initialize memory variables
    memory_p1 = tf.Variable(np.zeros(memory_depth), dtype=tf.float32)
    memory_p2 = tf.Variable(np.zeros(memory_depth), dtype=tf.float32)
    cycle_index = 0

    print("Predicting on test data...")

    for sample in range(samples):
        print(f"Sample {sample + 1}/{samples}")
        sample_predictions = []

        # Log start of new sample
        with open(log_filepath, 'a') as f:
            f.write(f"\n  Sample {sample + 1}/{samples}:\n")

        for i in range(len(encoded_phases)):
            time_step = i - cycle_index * memory_depth
            if time_step == memory_depth - 1:
                cycle_index += 1

            if stochastic:
                phase1_sample = np.random.normal(phase1.numpy(), var)
                phase3_sample = np.random.normal(phase3.numpy(), var)
                # Log sampled phases
                with open(log_filepath, 'a') as f:
                    f.write(" " * 4 + f"Step {i}: Phase1_sample = {phase1_sample:.4f}, "
                           f"Phase3_sample = {phase3_sample:.4f}\n")
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
    with open(log_filepath, 'a') as f:
        f.write("\nPrediction Summary:\n")
        f.write(" " * 2 + f"Number of test samples: {len(x_test)}\n")
        f.write(" " * 2 + f"Mean prediction: {np.mean(final_predictions):.4f}\n")
        if stochastic:
            f.write(" " * 2 + f"Mean predictive uncertainty: {np.mean(predictive_uncertainty):.4f}\n")
        f.write(" " * 2 + f"Mean absolute error: {np.mean(np.abs(final_predictions - targets)):.4f}\n")

    return final_predictions, targets, predictive_uncertainty