import numpy as np
import tensorflow as tf
import strawberryfields as sf
from strawberryfields.ops import *
import pickle
import random as rd
import matplotlib.pyplot as plt

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
    Computes the target output based on current and past inputs.

    Interpretation:
    - xt: Current input value (at time t).
    - xt1: Previous input value (at time t-1).
    - xt2: Input value before previous (at time t-2).

    This function defines the desired output for the model to learn.
    """
    return 0.4 * xt1 + 0.4 * xt1 * xt2 + 0.6 * xt ** 3 + 0.1

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

def train_memristor(x_train, dip, steps=50, learning_rate=0.003):
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
                prob_010 = tf.cast(tf.math.real(prob[0, 1, 0]), dtype=tf.float32)
                prob_001 = tf.cast(tf.math.real(prob[0, 0, 1]), dtype=tf.float32)

                # Update memory
                p1 = tf.tensor_scatter_nd_update(p1, [[t % dip]], [prob_010])
                p2 = tf.tensor_scatter_nd_update(p2, [[t % dip]], [prob_001])

                if phi >= 2:
                    # Compute the target function
                    f2 = target_function(x_train[phi], x_train[phi - 1], x_train[phi - 2])
                    f2 = tf.cast(f2, dtype=tf.float32)

                    # Compute loss
                    loss += tf.square(tf.abs(f2 - prob_001))

        # Compute gradients and update variables
        gradients = tape.gradient(loss, [phi1, phi3, x_2])
        opt.apply_gradients(zip(gradients, [phi1, phi3, x_2]))

        res_mem[('loss', 'tr', step)] = [loss.numpy(), phi1.numpy(), phi3.numpy(), x_2.numpy()]
        print(f"Loss at step {step + 1}: {loss.numpy()}")

    print(f"Final loss: {loss.numpy()}")
    print(f"Optimal parameters: phi1={phi1.numpy()}, phi3={phi3.numpy()}, x_2={x_2.numpy()}")
    return res_mem, phi1, phi3, x_2

def predict_memristor(x_test, dip, phi1, phi3, x_2):
    """
    Uses the trained memristor model to make predictions on test data.
    """
    eng = sf.Engine(backend="tf", backend_options={"cutoff_dim": 4})
    phienc = tf.constant(2 * np.arccos(x_test), dtype=tf.float32)

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
        prob_001 = tf.cast(tf.math.real(prob[0, 0, 1]), dtype=tf.float32)

        # Update memory
        prob_010 = tf.cast(tf.math.real(prob[0, 1, 0]), dtype=tf.float32)
        p1 = tf.tensor_scatter_nd_update(p1, [[t % dip]], [prob_010])
        p2 = tf.tensor_scatter_nd_update(p2, [[t % dip]], [prob_001])

        predictions.append(prob_001.numpy())

        if phi >= 2:
            # Compute the target function
            f2 = target_function(x_test[phi], x_test[phi - 1], x_test[phi - 2])
            targets.append(f2.numpy())

    return predictions, targets

def main():
    print("Memristor time lag")
    dip = 3  # Memory depth

    # Input data
    inp = np.random.random_sample(100) * 0.5  # Random values between 0 and 0.5
    x_train = tf.constant(np.sqrt(inp), dtype=tf.float32)

    # Train the memristor model
    res_mem, phi1, phi3, x_2 = train_memristor(x_train, dip)

    # Save training results
    with open("results_mem_t_lag_iris.pkl", "wb") as file:
        pickle.dump(res_mem, file)

    # Predict using the trained model
    x_test = x_train  # For simplicity, using the same data
    predictions, targets = predict_memristor(x_test, dip, phi1, phi3, x_2)

    # Print predictions and targets
    print("Predictions:", predictions)
    print("Targets:", targets)

    # Plotting
    # Since targets and predictions start from index 2 (after phi >= 2), we adjust the x-axis accordingly
    x_axis = np.arange(2, len(predictions))
    plt.figure(figsize=(10, 6))
    plt.plot(x_axis, predictions[2:], label='Predictions')
    plt.plot(x_axis, targets, label='Targets', linestyle='--')
    plt.xlabel('Data Point Index')
    plt.ylabel('Probability')
    plt.title('Memristor Model Predictions vs Targets')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()