#!/usr/bin/env python3

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import numpy as np
import tensorflow as tf
# Replace Strawberry Fields imports with Perceval
# import strawberryfields as sf
# from strawberryfields.ops import *
import perceval as pcvl
import perceval.components as pc
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

def build_circuit(phi_1, phi_2, phi_3, phi_enc):
    """
    Constructs the quantum circuit with the given parameters using Perceval.
    """
    circuit = pcvl.Circuit(3)

    # Input encoding MZI
    bs = pc.BS()
    circuit.add((0, 1), bs)
    circuit.add(1, pc.PS(phi_enc))
    circuit.add((0, 1), bs)

    # First MZI
    circuit.add((0, 1), bs)
    circuit.add(1, pc.PS(phi_1))
    circuit.add((0, 1), bs)

    # Memristor (Second MZI)
    circuit.add((1, 2), bs)
    circuit.add(1, pc.PS(phi_2))
    circuit.add((1, 2), bs)

    # Third MZI
    circuit.add((0, 1), bs)
    circuit.add(1, pc.PS(phi_3))
    circuit.add((0, 1), bs)

    return circuit

def train_memristor(x_train, y_train, memory_depth):
    """
    Trains the memristor model using Perceval.
    """
    # Initialize variables and optimizer
    phase1 = tf.Variable(tf.random.uniform([], 0, 2 * np.pi, dtype=tf.float64))
    phase3 = tf.Variable(tf.random.uniform([], 0, 2 * np.pi, dtype=tf.float64))
    memristor_weight = tf.Variable(tf.random.uniform([], 0, 1, dtype=tf.float64))

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    res_mem = {}

    encoded_phases = tf.constant(2 * np.arccos(x_train), dtype=tf.float64)
    num_samples = len(encoded_phases)

    # Initialize memory variables
    memory_p1 = tf.Variable(np.zeros(memory_depth), dtype=tf.float64)
    memory_p2 = tf.Variable(np.zeros(memory_depth), dtype=tf.float64)
    cycle_index = 0

    # Training loop
    for step in range(50):
        with tf.GradientTape() as tape:
            loss = 0.0

            for i in range(num_samples):
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

                circuit = build_circuit(phase1, memristor_phase, phase3, encoded_phases[i])

                # Simulate the circuit using Perceval
                simulator = pcvl.BackendFactory().get_backend("SLOS")
                # Create input state
                input_state = pcvl.BasicState([0, 1, 0])
                # Process the circuit
                probs = simulator.probabilities(circuit, input_state)

                # Get probabilities for specific states
                prob_state_001 = tf.cast(probs.get(pcvl.BasicState([0, 0, 1]), 0.0), dtype=tf.float64)
                prob_state_010 = tf.cast(probs.get(pcvl.BasicState([0, 1, 0]), 0.0), dtype=tf.float64)

                # Update memory variables
                index = [time_step % memory_depth]
                memory_p1 = tf.tensor_scatter_nd_update(memory_p1, [index], [prob_state_010])
                memory_p2 = tf.tensor_scatter_nd_update(memory_p2, [index], [prob_state_001])

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

def predict_memristor(x_test, y_test, memory_depth, phase1, phase3, memristor_weight):
    """
    Uses the trained memristor model to make predictions on test data using Perceval.
    """
    encoded_phases = tf.constant(2 * np.arccos(x_test), dtype=tf.float64)
    num_samples = len(encoded_phases)

    predictions = []
    targets = []

    # Initialize memory variables
    memory_p1 = tf.Variable(np.zeros(memory_depth), dtype=tf.float64)
    memory_p2 = tf.Variable(np.zeros(memory_depth), dtype=tf.float64)
    cycle_index = 0

    for i in range(num_samples):
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

        circuit = build_circuit(phase1, memristor_phase, phase3, encoded_phases[i])

        # Simulate the circuit using Perceval
        simulator = pcvl.BackendFactory().get_backend("SLOS")
        # Create input state
        input_state = pcvl.BasicState([0, 1, 0])
        # Process the circuit
        probs = simulator.probabilities(circuit, input_state)

        # Get probabilities for specific states
        prob_state_001 = tf.cast(probs.get(pcvl.BasicState([0, 0, 1]), 0.0), dtype=tf.float32)
        prob_state_010 = tf.cast(probs.get(pcvl.BasicState([0, 1, 0]), 0.0), dtype=tf.float32)

        # Update memory variables
        index = [time_step % memory_depth]
        memory_p1 = tf.tensor_scatter_nd_update(memory_p1, [index], [prob_state_010])
        memory_p2 = tf.tensor_scatter_nd_update(memory_p2, [index], [prob_state_001])

        # Store predictions and targets
        predictions.append(prob_state_001.numpy())
        targets.append(y_test[i])

    return predictions, targets

def main():
    X_train, y_train, X_test, y_test, _ = get_data(n_data=100, sigma_noise_1=0.0, datafunction=quartic_data)

    # Train the memristor model
    res_mem, phi1, phi3, x_2 = train_memristor(X_train, y_train, memory_depth=3)

    # Save training results
    with open("results_mem_t_lag_iris.pkl", "wb") as file:
        pickle.dump(res_mem, file)

    # Predict using the trained model
    predictions, targets = predict_memristor(X_test, y_test, memory_depth=3, phase1=phi1, phase3=phi3, memristor_weight=x_2)

    # Plotting the results
    plt.figure(figsize=(10, 6))
    plt.plot(X_test.numpy(), predictions, label='Predictions')
    plt.plot(X_test.numpy(), targets, label='Targets', linestyle='--')
    plt.xlabel('Input Data')
    plt.ylabel('Output')
    plt.title('Memristor Model Predictions vs Targets')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
