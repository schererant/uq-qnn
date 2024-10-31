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

def build_circuit(phi_1, phi_2, phi_3, phi_enc):
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
        Rgate(phi_enc)           | q[1]
        BSgate(np.pi/4, np.pi/2) | (q[0], q[1])
        
        # First MZI
        BSgate(np.pi/4, np.pi/2) | (q[0], q[1])
        Rgate(phi_1)             | q[1]
        BSgate(np.pi/4, np.pi/2) | (q[0], q[1])
        
        # Memristor (Second MZI)
        BSgate(np.pi/4, np.pi/2) | (q[1], q[2])
        Rgate(phi_2)             | q[1]
        BSgate(np.pi/4, np.pi/2) | (q[1], q[2])
        
        # Third MZI
        BSgate(np.pi/4, np.pi/2) | (q[0], q[1])
        Rgate(phi_3)             | q[1]
        BSgate(np.pi/4, np.pi/2) | (q[0], q[1])
    return circuit

def train_memristor(x_train, y_train, dip, steps=50, learning_rate=0.1):
    """
    Trains the memristor model using the provided training data.
    """
    res_mem = {}
    eng = sf.Engine(backend="tf", backend_options={"cutoff_dim": 4})

    # Initialize variables
    phi1 = tf.Variable(rd.uniform(0.01, 1) * 2 * np.pi, dtype=tf.float32,
                       constraint=lambda z: tf.clip_by_value(z, 0, 2 * np.pi))
    phi3 = tf.Variable(rd.uniform(0.01, 1) * 2 * np.pi, dtype=tf.float32,
                       constraint=lambda z: tf.clip_by_value(z, 0, 2 * np.pi))
    x_2 = tf.Variable(rd.uniform(0.01, 1), dtype=tf.float32,
                      constraint=lambda z: tf.clip_by_value(z, 0.01, 1))  # Memristor parameter

    phienc = tf.constant(2 * np.arccos(x_train), dtype=tf.float32)

    print("Training memristor model")
    print(f"Initial parameters: phi1={phi1.numpy()}, phi3={phi3.numpy()}, x_2={x_2.numpy()}")

    # Optimizer
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    for step in range(steps):
        # Reset the engine if it has already been executed
        if eng.run_progs:
            eng.reset()

        with tf.GradientTape() as tape:
            loss = 0
            index = 0
            p1 = tf.Variable(np.zeros(dip), dtype=tf.float32)
            p2 = tf.Variable(np.zeros(dip), dtype=tf.float32)

            for phi in range(len(phienc)):
                t = phi - index * dip
                if t == dip - 1:
                    index += 1

                if phi == 0:
                    # Empty memory, start with initial value
                    phi_2_value = tf.acos(tf.sqrt(0.5))
                    circuit = build_circuit(phi1, phi_2_value, phi3, phienc[phi])
                    results = eng.run(circuit)
                else:
                    # Use previous memory values
                    mem_value = tf.acos(tf.sqrt(tf.reduce_sum(p1) / dip + x_2 * tf.reduce_sum(p2) / dip))
                    circuit = build_circuit(phi1, mem_value, phi3, phienc[phi])
                    results = eng.run(circuit)

                # Get probabilities
                prob = results.state.all_fock_probs()

                # Extract probabilities and cast to float32
                prob_010 = tf.cast(tf.math.real(prob[0, 1, 0]), dtype=tf.double)
                prob_001 = tf.cast(tf.math.real(prob[0, 0, 1]), dtype=tf.double)
        

                # Update memory
                p1 = tf.tensor_scatter_nd_update(p1, [[t % dip]], [prob_010])
                p2 = tf.tensor_scatter_nd_update(p2, [[t % dip]], [prob_001])


                # print(y_train[phi])
                # print(prob_001)
                loss += tf.square(tf.abs(y_train[phi] - prob_001))

                # if phi >= 2:
                #     # Compute the target function
                #     f2 = target_function(x_train[phi], x_train[phi - 1], x_train[phi - 2])
                #     f2 = tf.cast(f2, dtype=tf.float32)

                #     f2 = y_train

                #     # Compute loss
                #     loss += tf.square(tf.abs(f2 - prob_001))

        # Compute gradients and update variables
        gradients = tape.gradient(loss, [phi1, phi3, x_2])
        opt.apply_gradients(zip(gradients, [phi1, phi3, x_2]))

        res_mem[('loss', 'tr', step)] = [loss.numpy(), phi1.numpy(), phi3.numpy(), x_2.numpy()]
        print(f"Loss at step {step + 1}: {loss.numpy()}")

    print(f"Final loss: {loss.numpy()}")
    print(f"Optimal parameters: phi1={phi1.numpy()}, phi3={phi3.numpy()}, x_2={x_2.numpy()}")
    return res_mem, phi1, phi3, x_2

def predict_memristor(x_test, y_test, dip, phi1, phi3, x_2):
    """
    Uses the trained memristor model to make predictions on test data.
    """
    eng = sf.Engine(backend="tf", backend_options={"cutoff_dim": 4})
    phienc = tf.constant(2 * np.arccos(x_test), dtype=tf.float64)

    predictions = []
    targets = []

    p1 = tf.Variable(np.zeros(dip), dtype=tf.float32)
    p2 = tf.Variable(np.zeros(dip), dtype=tf.float32)
    index = 0

    for phi in range(len(phienc)):
        t = phi - index * dip
        if t == dip - 1:
            index += 1

        if phi == 0:
            phi_2_value = tf.acos(tf.sqrt(0.5))
            circuit = build_circuit(phi1, phi_2_value, phi3, phienc[phi])
            results = eng.run(circuit)
        else:
            mem_value = tf.acos(tf.sqrt(tf.reduce_sum(p1) / dip + x_2 * tf.reduce_sum(p2) / dip))
            circuit = build_circuit(phi1, mem_value, phi3, phienc[phi])
            results = eng.run(circuit)

        prob = results.state.all_fock_probs()

        prob_001 = tf.cast(tf.math.real(prob[0, 0, 1]), dtype=tf.float32)

        prob_010 = tf.cast(tf.math.real(prob[0, 1, 0]), dtype=tf.float32)
        p1 = tf.tensor_scatter_nd_update(p1, [[t % dip]], [prob_010])
        p2 = tf.tensor_scatter_nd_update(p2, [[t % dip]], [prob_001])

        predictions.append(prob_001.numpy())
        targets.append(y_test[phi])
    return predictions, targets

def main():
    print("Memristor time lag")
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
    plt.plot(X_test, predictions, label='Predictions')
    plt.plot(X_test, targets, label='Targets', linestyle='--')
    plt.xlabel('Input Data')
    plt.ylabel('Output')
    plt.title('Memristor Model Predictions vs Targets')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()