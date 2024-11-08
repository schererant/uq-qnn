#!/usr/bin/env python3

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import numpy as np
import tensorflow as tf
import strawberryfields as sf
from strawberryfields.ops import *
import pickle
import random as rd
import matplotlib.pyplot as plt
import warnings

from collections.abc import Callable

tf.get_logger().setLevel('ERROR')
warnings.filterwarnings("ignore")

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
rd.seed(42)

def quartic_data(input_data):
    """ Create labels with quartic function.
    
    Args:
        input_data: tf array
    
    Returns:
        y: quartic function applied to input array
    """

    y = tf.convert_to_tensor(tf.pow(input_data, 4))

    return y


def get_data(n_data: int =100, sigma_noise_1: float = 0.0, datafunction: Callable = quartic_data):
    """Define a function based toy regression dataset.

    Args:
      n_data: number of data points
      sigma_noise_1: injected sigma noise on targets
      datafunction: function to compute labels based on input data

    Returns:
      train_input, train_target, test_input, test_target
    """
    x_min = 0
    x_max = 1
    X_train = tf.linspace(x_min, x_max, n_data)
    
    # split training set
    gap_start = x_min + 0.35 * (x_max - x_min)
    gap_end = x_min + 0.6 * (x_max - x_min)

    # create label noise
    noise_1 = tf.random.normal([n_data], 0, 1, tf.float32, seed=1) * sigma_noise_1
    noise_1 = tf.where(X_train > gap_end, 0.0, noise_1)  # Only add noise to the left

 
    # create simple function based labels data set and
    # add gaussian noise
    label_noise = noise_1
    y_train = datafunction(X_train) # + label_noise

    #:TODO @nina: why do we need this?
    train_idx = (X_train < gap_end) & (X_train > gap_start)

    # update X_train
    X_train = X_train[~train_idx]
    y_train = y_train[~train_idx]

    # test over the whole line
    X_test = tf.linspace(x_min + x_min * 0.1, x_max - x_max * 0.1, 500)
    y_test = datafunction(X_test)


    return X_train, y_train, X_test, y_test, label_noise

def plot_toy_data(X_train, y_train, X_test, y_test):
    """Plot the toy data."""
    fig, ax = plt.subplots(1)
    ax.scatter(X_train, y_train, color="blue", label="train_data")
    ax.scatter(X_test, y_test, color="orange", label="test_data")
    plt.legend()
    plt.show()

def plot_predictions(
    X_train, y_train, X_test, y_test, y_pred, pred_std=None, pred_quantiles=None, epistemic=None, aleatoric=None, title=None
) -> None:
    """Plot predictive uncertainty as well as epistemic and aleatoric separately.
    
    Args:
      X_train:
      y_train:
      X_test:
      y_test:
      y_pred:
      pred_std:
      pred_quantiles:
      epistemic: for us this is predictive_uncertainty
      aleatoric:
    """
    # fig, ax = plt.subplots(ncols=2)
    fig = plt.figure()
    ax0 = fig.add_subplot(1, 2, 1)

    # model predictive uncertainty bands on the left
    ax0.scatter(X_test, y_test, color="gray", label="ground truth", s=0.5)
    ax0.scatter(X_train, y_train, color="blue", label="train_data")
    ax0.scatter(X_test, y_pred, color="orange", label="predictions")

    if pred_std is not None:
        ax0.fill_between(
            X_test.squeeze(),
            y_pred - pred_std,
            y_pred + pred_std,
            alpha=0.3,
            color="tab:red",
            label="$\sqrt{\mathbb{V}\,[y]}$",
        )

    if pred_quantiles is not None:
        ax0.plot(X_test, pred_quantiles, color="tab:red", linestyle="--", label="quantiles")

    if title is not None:
        ax0.set_title(title + " showing mean +- std")

    # epistemic and aleatoric uncertainty plots on right
    # epistemic uncertainty figure
    ax1 = fig.add_subplot(2, 2, 2)
    if epistemic is not None:
      ax1.scatter(X_test, y_test, color="gray", label="ground truth", s=0.5)
      ax1.set_title("Epistemic Uncertainty")
      ax1.fill_between(
            X_test.squeeze(),
            y_pred - epistemic,
            y_pred + epistemic,
            alpha=0.3,
            color="tab:red",
            label="Epistemic",
        )
      ax1.set_title("Epistemic Uncertainty")
      ax1.legend()
    else:
      ax1.text(0.5, 0.5, "This Method does not quantify epistemic uncertainty.", horizontalalignment='center', verticalalignment='center', fontsize=15)

    # aleatoric uncertainty figure
    ax2 = fig.add_subplot(2, 2, 4)
    if aleatoric is not None:
      ax2.scatter(X_test, y_test, color="gray", label="ground truth", s=0.5)
      ax2.fill_between(
            X_test.squeeze(),
            y_pred - aleatoric,
            y_pred + aleatoric,
            alpha=0.3,
            color="tab:red",
            label="Aleatoric",
        )
      ax2.set_title("Aleatoric Uncertainty")
    else:
      ax2.text(0.5, 0.5, "This Method does not quantify aleatoric uncertainty.", horizontalalignment='center', verticalalignment='center', fontsize=15)
    
    ax0.legend()
    plt.show()

def memristor_update_function(x, y1, y2):
    """
    Computes the memristor update based on current input x and past values y1 and y2.

    Interpretation:
    - x: Current input value.
    - y1: Previous output (at time t-1).
    - y2: Output before previous (at time t-2).

    This function models how the memristor's state changes over time,
    incorporating both current input and past outputs.
    """
    return 0.4 * y1 + 0.4 * y1 * y2 + 0.6 * x ** 3 + 0.1

def multiply_three_inputs(x1, x2, x3):
    """
    Multiplies three input values.

    Interpretation:
    - x1, x2, x3: Input values at times t, t-1, and t-2, respectively.

    This function models a target where the output is the product of three inputs.
    """
    return x1 * x2 * x3

def target_function(xt, xt1, xt2):
    """
    Computes the target output as a sinusoidal function based on current and past inputs.

    Interpretation:
    - xt: Current input value at time t.
    - xt1: Previous input value at time t-1.
    - xt2: Input value at time t-2.

    This function defines a smooth, sinusoidal target for the model to learn.
    """
    return np.sin(2 * np.pi * (xt + xt1 + xt2)) + 1

# def target_function(xt, xt1, xt2):
#     """
#     Computes the target output based on current and past inputs.

#     Interpretation:
#     - xt: Current input value (at time t).
#     - xt1: Previous input value (at time t-1).
#     - xt2: Input value before previous (at time t-2).

#     This function defines the desired output for the model to learn.
#     """
#     return 0.4 * xt1 + 0.4 * xt1 * xt2 + 0.6 * xt ** 3 + 0.1

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

def train_memristor(x_train, y_train, memory_depth, training_steps=1):
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

    phase1 = tf.Variable(rd.uniform(0.01, 1) * 2 * np.pi, dtype=tf.float32,
                       constraint=lambda z: tf.clip_by_value(z, 0, 2 * np.pi))
    phase3 = tf.Variable(rd.uniform(0.01, 1) * 2 * np.pi, dtype=tf.float32,
                       constraint=lambda z: tf.clip_by_value(z, 0, 2 * np.pi))
    memristor_weight = tf.Variable(rd.uniform(0.01, 1), dtype=tf.float32,
                      constraint=lambda z: tf.clip_by_value(z, 0.01, 1))  # Memristor parameter

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)
    eng = sf.Engine(backend="tf", backend_options={"cutoff_dim": 4})
    res_mem = {}

    encoded_phases = tf.constant(2 * np.arccos(x_train), dtype=tf.float32)
    num_samples = len(encoded_phases)

    # Initialize memory variables
    memory_p1 = tf.Variable(np.zeros(memory_depth), dtype=tf.float32)
    memory_p2 = tf.Variable(np.zeros(memory_depth), dtype=tf.float32)
    cycle_index = 0

    # Training loop
    for step in range(training_steps):
        # Reset the engine if it has already been executed
        if eng.run_progs:
            eng.reset()

        with tf.GradientTape() as tape:
            loss = 0.0

            for i in range(num_samples):
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

            res_mem[('loss', 'tr', step)] = [loss.numpy(), phase1.numpy(), phase3.numpy(), memristor_weight.numpy()]
            print(f"Loss at step {step + 1}: {loss.numpy()}")

    print(f"Final loss: {loss.numpy()}")
    print(f"Optimal parameters: phase1={phase1.numpy()}, phase3={phase3.numpy()}, memristor_weight={memristor_weight.numpy()}")

    return res_mem, phase1, phase3, memristor_weight

def predict_memristor(
    x_test,
    y_test,
    memory_depth,
    phase1,
    phase3,
    memristor_weight,
    stochastic: bool = False,
    samples: int = 20,
    var: float = 0.1,
):
    """
    Uses the trained memristor model to make predictions on test data.

    Args:
        x_test: Test input data.
        y_test: Test target data.
        memory_depth: Memory depth of the memristor.
        phase1: Trained phase parameter 1.
        phase3: Trained phase parameter 3.
        memristor_weight: Trained weight parameter for the memristor update function.
        stochastic: If True, samples `phase1` and `phase3` from normal distributions
            to compute predictive uncertainty.
        samples: Number of samples to draw if `stochastic` is True.
        var: Variance for the normal distribution used when `stochastic` is True.

    Returns:
        If `stochastic` is True:
            predictions_mean: Mean of predictions over samples.
            predictive_uncertainty: Standard deviation of predictions over samples.
            targets: Test target data.
        Else:
            predictions: Predictions on test data.
            targets: Test target data.
    """
    eng = sf.Engine(backend="tf", backend_options={"cutoff_dim": 4})
    encoded_phases = tf.constant(2 * np.arccos(x_test), dtype=tf.float64)

    # Initialize memory variables for each sample
    memory_p1 = tf.Variable(np.zeros(memory_depth), dtype=tf.float32)
    memory_p2 = tf.Variable(np.zeros(memory_depth), dtype=tf.float32)

    if stochastic:
        predictions = np.zeros((samples, len(encoded_phases)))
        targets = y_test.numpy()

        for s in range(samples):
            # Sample phase1_sample and phase3_sample from normal distributions
            phase1_sample = np.random.normal(phase1.numpy(), var)
            phase3_sample = np.random.normal(phase3.numpy(), var)

            
            cycle_index = 0

            for i in range(len(encoded_phases)):
                time_step = i - cycle_index * memory_depth
                if time_step == memory_depth - 1:
                    cycle_index += 1

                if i == 0:
                    memristor_phase = tf.acos(tf.sqrt(0.5))
                else:
                    memristor_phase = tf.acos(
                        tf.sqrt(
                            tf.reduce_sum(memory_p1) / memory_depth
                            + memristor_weight * tf.reduce_sum(memory_p2) / memory_depth
                        )
                    )

                # Build and run the circuit with sampled phases
                circuit = build_circuit(
                    phase1_sample, memristor_phase, phase3_sample, encoded_phases[i]
                )
                results = eng.run(circuit)

                # Get probabilities from the circuit results
                prob = results.state.all_fock_probs()
                prob_state_001 = tf.cast(tf.math.real(prob[0, 0, 1]), dtype=tf.float32)
                prob_state_010 = tf.cast(tf.math.real(prob[0, 1, 0]), dtype=tf.float32)

                # Update memory variables
                memory_p1 = tf.tensor_scatter_nd_update(
                    memory_p1, [[time_step % memory_depth]], [prob_state_010]
                )
                memory_p2 = tf.tensor_scatter_nd_update(
                    memory_p2, [[time_step % memory_depth]], [prob_state_001]
                )

                predictions[s, i] = prob_state_001.numpy()

                # Reset the engine for the next run
                if eng.run_progs:
                    eng.reset()

        # Compute mean and standard deviation over samples for each test input
        predictions_mean = np.mean(predictions, axis=0)
        predictive_uncertainty = np.std(predictions, axis=0)

        return predictions_mean, predictive_uncertainty, targets

    else:
        # Initialize lists to store predictions and targets
        predictions = np.zeros((samples, len(encoded_phases)))
        targets = y_test.numpy()

        # Initialize memory variables
        memory_p1 = tf.Variable(np.zeros(memory_depth), dtype=tf.float32)
        memory_p2 = tf.Variable(np.zeros(memory_depth), dtype=tf.float32)
        cycle_index = 0

        for i in range(len(encoded_phases)):
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
            prob_state_001 = tf.cast(tf.math.real(prob[0, 0, 1]), dtype=tf.float32)
            prob_state_010 = tf.cast(tf.math.real(prob[0, 1, 0]), dtype=tf.float32)

            # Update memory variables
            memory_p1 = tf.tensor_scatter_nd_update(memory_p1, [[time_step % memory_depth]], [prob_state_010])
            memory_p2 = tf.tensor_scatter_nd_update(memory_p2, [[time_step % memory_depth]], [prob_state_001])

            predictions.append(prob_state_001.numpy())
            targets.append(y_test[i])
            
            return predictions, targets

def main():
    print("Training the memristor model...")
    dip = 3  # Memory depth

    # Generate data using get_data function
    X_train, y_train, X_test, y_test, _ = get_data(n_data=100, sigma_noise_1=0.0, datafunction=quartic_data)

    # Train the memristor model
    res_mem, phi1, phi3, x_2 = train_memristor(X_train, y_train, dip)

    # Save training results
    with open("results_mem_t_lag_iris.pkl", "wb") as file:
        pickle.dump(res_mem, file)

    # Predict using the trained model
    predictions, targets = predict_memristor(X_test, y_test, dip, phi1, phi3, x_2)

    # Plotting the results
    plt.figure(figsize=(10, 6))
    plt.plot(X_test.numpy() , predictions, label='Predictions')
    plt.plot(X_test.numpy() , targets, label='Targets', linestyle='--')
    plt.xlabel('Input Data')
    plt.ylabel('Output')
    plt.title('Memristor Model Predictions vs Targets')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()









# Trainings + Test Daten plotten
# Visuelle auswertung
# Quantitative Auswertung

# Selective prediction @Nina


# Perceval benutzen

# Pro run 