import numpy as np
import tensorflow as tf
import strawberryfields as sf
from strawberryfields.ops import *
import random as rd
from datetime import datetime
from utils import log_training_loss




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

def train_memristor(x_train, y_train, memory_depth, training_steps, learning_rate = 0.1, cutoff_dim = 4, filename = None, logger = None):
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

# def predict_memristor(x_test, y_test, memory_depth, phase1, phase3, memristor_weight, stochastic: bool = True, samples: int = 1, var: float = 0.1, cutoff_dim = 4, filename = None):
#     """
#     Uses the trained memristor model to make predictions on test data.
#     """
#     # for UQ stuff we want to have the circuit below with the memristor parametrized by memristor_weight and then we want to make the parameters
#     # for the MZIs phase1 and phase3 stochastic
#     # that means sample phase1_sample in np.normal(phase1, var)  and phase3_sample in np.normal(phase3, var)
#     # where we experiment with the var>0 and phase1 and phase3 are obtained from the training loop below   

#     # if clause with stochastic == True
#     # samples is number of phase1_sample in np.normal(phase1, var)
#     # also possible phase1_sample, phase3_sample in np.normal([phase1,phase3], [var,var])
#     # else blow as it is

#     # eval predictions: stack of predictions mean over samples for each phi in len(phienc)
#     # predictive_uncertainty: prediction std over samples for each phi in len(phienc)

#     # # Create log file
#     # log_filepath = log_prediction_results(x_test, y_test, memory_depth, phase1, phase3, memristor_weight, stochastic, samples, var, cutoff_dim)
    

#     eng = sf.Engine(backend="tf", backend_options={"cutoff_dim": cutoff_dim})
#     encoded_phases = tf.constant(2 * np.arccos(x_test), dtype=tf.float64)

#     # Initialize lists to store predictions and targets
#     all_predictions = []
#     targets = []

#     # Initialize memory variables
#     memory_p1 = tf.Variable(np.zeros(memory_depth), dtype=tf.float32)
#     memory_p2 = tf.Variable(np.zeros(memory_depth), dtype=tf.float32)
#     cycle_index = 0

#     print("Predicting on test data...")

#     for sample in range(samples):
#         print(f"Sample {sample + 1}/{samples}")
#         sample_predictions = []

#         for i in range(len(encoded_phases)):
#             time_step = i - cycle_index * memory_depth
#             if time_step == memory_depth - 1:
#                 cycle_index += 1

#             if stochastic:
#                 phase1_sample = np.random.normal(phase1.numpy(), var)
#                 phase3_sample = np.random.normal(phase3.numpy(), var)
#             else:
#                 phase1_sample = phase1.numpy()
#                 phase3_sample = phase3.numpy()


#             if i == 0:
#                 memristor_phase = tf.acos(tf.sqrt(0.5))
#             else:
#                 memristor_phase = tf.acos(tf.sqrt(
#                     tf.reduce_sum(memory_p1) / memory_depth +
#                     memristor_weight * tf.reduce_sum(memory_p2) / memory_depth
#                 ))

#             circuit = build_circuit(phase1_sample, memristor_phase, phase3_sample, encoded_phases[i])
#             results = eng.run(circuit)

#             # Get probabilities from the circuit results
#             prob = results.state.all_fock_probs()
#             prob_state_001 = tf.cast(tf.math.real(prob[0, 0, 1]), dtype=tf.float32)
#             prob_state_010 = tf.cast(tf.math.real(prob[0, 1, 0]), dtype=tf.float32)

#             # Update memory variables
#             memory_p1 = tf.tensor_scatter_nd_update(memory_p1, [[time_step % memory_depth]], [prob_state_010])
#             memory_p2 = tf.tensor_scatter_nd_update(memory_p2, [[time_step % memory_depth]], [prob_state_001])

#             sample_predictions.append(prob_state_001.numpy())
#             if sample == 0:
#                 targets.append(float(y_test[i].numpy()))

#         all_predictions.append(sample_predictions)

#     # Convert all_predictions to a NumPy array for easier manipulation
#     all_predictions = np.array(all_predictions)

#     if stochastic:
#         # Calculate mean and standard deviation along the column axis
#         final_predictions = np.mean(all_predictions, axis=0)
#         predictive_uncertainty = np.std(all_predictions, axis=0)
#     else:
#         final_predictions = all_predictions[0]
#         predictive_uncertainty = np.array([])
#         targets = np.array(targets)



#     return final_predictions, targets, predictive_uncertainty