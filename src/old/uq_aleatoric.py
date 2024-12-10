import numpy as np
import tensorflow as tf
import strawberryfields as sf
import random as rd
from model import build_circuit

def train_memristor_aleatoric(x_train, y_train, memory_depth, training_steps=100):
    """
    Trains the aleatoric memristor model using the training data - ideally use training data with noise.

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
    # epsilon for numerical stability of the NLL loss function

    epsilon = 0.001

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

                # Compute the NLL loss which is now the negative log-likelihood of a 1d Gaussian
                loss += tf.square(tf.abs(y_train[i] - prob_state_001)) * tf.math.divide(1.0,tf.pow(prob_state_010,2)*2+epsilon) + tf.math.log(prob_state_010+epsilon)

            # Compute gradients and update variables
            gradients = tape.gradient(loss, [phase1, phase3, memristor_weight])
            optimizer.apply_gradients(zip(gradients, [phase1, phase3, memristor_weight]))

            res_mem[('loss', 'tr', step)] = [loss.numpy(), phase1.numpy(), phase3.numpy(), memristor_weight.numpy()]
            print(f"Loss at step {step + 1}: {loss.numpy()}")

    print(f"Final loss: {loss.numpy()}")
    print(f"Optimal parameters: phase1={phase1.numpy()}, phase3={phase3.numpy()}, memristor_weight={memristor_weight.numpy()}")

    return res_mem, phase1, phase3, memristor_weight

def predict_memristor_aleatoric(x_test, y_test, memory_depth, phase1, phase3, memristor_weight, stochastic: bool = True, samples: int = 20, var: float = 0.1):
    """
    Uses the trained aleatoric memristor model to make predictions on test data.
    """
    # for UQ stuff we want to have the circuit below with the memristor parametrized by memristor_weight and then we want to make the parameters
    # for the MZIs phase1 and phase3 stochastic
    # that means sample phase1_sample in np.normal(phase1, var)  and phase3_sample in np.normal(phase3, var)
    # where we experiment with the var>0 and phase1 and phase3 are obtained from the training loop below   

    # if clause with stochastic == True
    # samples is number of phase1_sample in np.normal(phase1, var)
    # also possible phase1_sample, phase3_sample in np.normal([phase1,phase3], [var,var])
    # else blow as it is

    # eval predictions: stack of predictions mean over samples for each phi in len(phienc)
    # predictive_uncertainty: prediction std over samples for each phi in len(phienc)

    eng = sf.Engine(backend="tf", backend_options={"cutoff_dim": 4})
    encoded_phases = tf.constant(2 * np.arccos(x_test), dtype=tf.float64)

    # Initialize lists to store predictions and targets
    all_predictions = []
    # second dimension of predictions with aleatoric uncertainty prediction
    all_sigmas = []
    targets = []

    # Initialize memory variables
    memory_p1 = tf.Variable(np.zeros(memory_depth), dtype=tf.float32)
    memory_p2 = tf.Variable(np.zeros(memory_depth), dtype=tf.float32)
    cycle_index = 0

    print("Predicting on test data...")

    for sample in range(samples):
        print(f"Sample {sample + 1}/{samples}")
        sample_predictions = []
        sample_sigmas = []

        for i in range(len(encoded_phases)):
            time_step = i - cycle_index * memory_depth
            if time_step == memory_depth - 1:
                cycle_index += 1

            if stochastic:
                phase1_sample = np.random.normal(phase1.numpy(), var)
                phase3_sample = np.random.normal(phase3.numpy(), var)
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
            sample_sigmas.append(prob_state_010.numpy())
            if sample == 0:
                targets.append(y_test[i])

        all_predictions.append(sample_predictions)
        all_sigmas.append(sample_sigmas)

    # Convert all_predictions to a NumPy array for easier manipulation
    all_predictions = np.array(all_predictions)
    all_sigmas = np.array(all_sigmas)

    if stochastic:
        # Calculate mean and standard deviation along the column axis
        final_predictions = np.mean(all_predictions, axis=0)
        predictive_uncertainty = np.std(all_predictions, axis=0)
        aleatoric_uncertainty = np.mean(all_sigmas, axis =0)
    else:
        final_predictions = all_predictions[0]
        predictive_uncertainty = np.zeros_like(final_predictions)
        all_sigmas = all_sigmas[0]


    return final_predictions, targets, predictive_uncertainty, aleatoric_uncertainty