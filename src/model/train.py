from datetime import datetime
import tensorflow as tf
import numpy as np
import random as rd
import strawberryfields as sf

from ..quantum.memristor_chip import build_circuit

def train_memristor(x_train, y_train, memory_depth, training_steps, learning_rate = 0.1, cutoff_dim = 4, filename = None):
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
    log_filepath = filename

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
    phase1 = tf.Variable(rd.uniform(0.01, 1) * 2 * np.pi, dtype=tf.float32,
                       constraint=lambda z: tf.clip_by_value(z, 0, 2 * np.pi))
    phase3 = tf.Variable(rd.uniform(0.01, 1) * 2 * np.pi, dtype=tf.float32,
                       constraint=lambda z: tf.clip_by_value(z, 0, 2 * np.pi))
    memristor_weight = tf.Variable(rd.uniform(0.01, 1), dtype=tf.float32,
                      constraint=lambda z: tf.clip_by_value(z, 0.01, 1))  # Memristor parameter
    
    # Write header
    with open(log_filepath, 'a') as f:
        f.write("\nInitial Parameters:\n")
        f.write(" " * 2 + f"Phase1: {float(phase1):.4f}\n")
        f.write(" " * 2 + f"Phase3: {float(phase3):.4f}\n")
        f.write(" " * 2 + f"Memristor Weight: {float(memristor_weight):.4f}\n")
        f.write("\nTraining Progress:\n")

        
        
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    eng = sf.Engine(backend="tf", backend_options={"cutoff_dim": cutoff_dim})
    res_mem = {}

    encoded_phases = tf.constant(2 * np.arccos(x_train), dtype=tf.float64)
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

            # Log results every step
            # log_training_loss(log_filepath, step, loss, phase1, phase3, memristor_weight)
            with open(log_filepath, 'a') as f:
                f.write(" " * 2 + f"Step {step:4d}: Loss = {loss:.4f}, "
                    f"Phase1 = {float(phase1):.4f}, "
                    f"Phase3 = {float(phase3):.4f}, "
                    f"Weight = {float(memristor_weight):.4f}\n")


            res_mem[('loss', 'tr', step)] = [loss.numpy(), phase1.numpy(), phase3.numpy(), memristor_weight.numpy()]
            print(f"Loss at step {step + 1}: {loss.numpy()}")

    print(f"Final loss: {loss.numpy()}")
    print(f"Optimal parameters: phase1={phase1.numpy()}, phase3={phase3.numpy()}, memristor_weight={memristor_weight.numpy()}")

    with open(log_filepath, 'a') as f:
        f.write("\nFinal Parameters:\n") 
        f.write(" " * 2 + f"Phase1: {float(phase1):.4f}\n")
        f.write(" " * 2 + f"Phase3: {float(phase3):.4f}\n")
        f.write(" " * 2 + f"Memristor Weight: {float(memristor_weight):.4f}\n")
        f.write("\nTraining Summary:\n")
        f.write(" " * 2 + f"Initial Loss: {res_mem[('loss', 'tr', 0)][0]:.4f}\n")
        f.write(" " * 2 + f"Final Loss: {loss:.4f}\n")
    
    return res_mem, phase1, phase3, memristor_weight