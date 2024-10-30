import strawberryfields as sf
import numpy as np
import tensorflow as tf
import random as rd
import pickle

# Function to initialize random phases and parameters
def initialize_parameters():
    phi1 = tf.Variable(rd.uniform(0.01, 1) * 2 * np.pi, constraint=lambda z: tf.clip_by_value(z, 0, 2 * np.pi))
    phi2 = tf.Variable(rd.uniform(0.01, 1), constraint=lambda z: tf.clip_by_value(z, 0.01, 1))  # Memoristor phase
    phi3 = tf.Variable(rd.uniform(0.01, 1) * 2 * np.pi, constraint=lambda z: tf.clip_by_value(z, 0, 2 * np.pi))
    return phi1, phi2, phi3

# Function to create the quantum circuit
def create_circuit(circuit, phi1, phi2, phi3, phienc):
    with circuit.context as q:
        Vac | q[0]
        Fock(1) | q[1]
        Vac | q[2]
        
        # Encoding and first MZI
        BSgate(np.pi / 4, np.pi / 2) | (q[0], q[1])
        Rgate(phienc) | q[1]  # Encoding
        BSgate(np.pi / 4, np.pi / 2) | (q[0], q[1])
        
        # First MZI
        BSgate(np.pi / 4, np.pi / 2) | (q[0], q[1])
        Rgate(phi1) | q[1]  # State preparation
        BSgate(np.pi / 4, np.pi / 2) | (q[0], q[1])

        # Second MZI (Memoristor)
        BSgate(np.pi / 4, np.pi / 2) | (q[1], q[2])
        Rgate(phi2) | q[1]  # Memoristor phase
        BSgate(np.pi / 4, np.pi / 2) | (q[1], q[2])

        # Third MZI
        BSgate(np.pi / 4, np.pi / 2) | (q[0], q[1])
        Rgate(phi3) | q[1]  # State tomography
        BSgate(np.pi / 4, np.pi / 2) | (q[0], q[1])
    return circuit

# Function to perform training
def train_memristor_model(circuit, eng, phienc, phi1, phi2, phi3, steps, dip, x_train, n_samples=20, sigma1=0.1, sigma3=0.1):
    sample1 = np.random.normal(phi1.value(), sigma1, n_samples)
    sample3 = np.random.normal(phi3.value(), sigma3, n_samples)
    
    for step in range(steps):
        if eng.run_progs:
            eng.reset()

        with tf.GradientTape() as tape:
            loss = 0
            p1 = np.zeros(dip)  # Memory placeholder
            p2 = np.zeros(dip)

            for phi in range(len(phienc)):
                t = phi % dip  # Time lag index
                if phi == 0:  # First step, initialize
                    while (phi1, phi3) in (sample1, sample3):
                        results = eng.run(circuit, args={"phi1": phi1, "phi3": phi3, "phi2": tf.Variable(tf.acos(np.sqrt(0.5))), "phienc": phienc[phi]})
                else:
                    while (phi1, phi3) in (sample1, sample3):
                        results = eng.run(circuit, args={"phi1": phi1, "phi3": phi3, "phi2": tf.acos(tf.sqrt(np.sum(p1) / dip + phi2 * np.sum(p2) / dip)), "phienc": phienc[phi]})

                prob = results.state.all_fock_probs()
                prob_mean = prob.mean()

                # Store probabilities in memory
                p1[t] = tf.Variable(np.real(prob_mean[0, 1, 0]))
                p2[t] = tf.Variable(np.real(prob_mean[0, 0, 1]))

                # Prediction with memory and loss calculation
                if phi >= 2:
                    f2 = function_lag_iris(x_train[phi]**2, x_train[phi-1]**2, x_train[phi-2]**2)
                    loss += (abs(f2 - prob_mean[0, 0, 1])) ** 2

            # Compute and apply gradients
            gradients = tape.gradient(loss, [phi1, phi3, phi2])
            opt.apply_gradients(zip(gradients, [phi1, phi3, phi2]))

# Main function to run the memristor time lag prediction
def run_memristor_time_lag():
    print("Memristor Time Lag")
    prove_num = 1
    power = 3  # Cubic function prediction
    res_mem = {}
    
    for p in range(prove_num):
        eng = sf.Engine(backend="tf", backend_options={"cutoff_dim": 4})
        circuit = sf.Program(3)
        
        # Use random training points
        x_train = np.sqrt(inp)
        phienc = 2 * np.arccos(x_train)
        
        print(f"Test {p + 1} of {prove_num}")
        
        # Initialize random parameters for the circuit
        phi1, phi2, phi3 = initialize_parameters()
        
        # Create quantum circuit with initialized parameters
        circuit = create_circuit(circuit, phi1, phi2, phi3, phienc)
        
        # Train the model
        steps = 1  # Number of training steps
        train_memristor_model(circuit, eng, phienc, phi1, phi2, phi3, steps, power, x_train)

    # Save results
    with open(f"results_mem_{power}_time_lag.pickle", "wb") as file:
        pickle.dump(res_mem, file)

# Run the main function
run_memristor_time_lag()


