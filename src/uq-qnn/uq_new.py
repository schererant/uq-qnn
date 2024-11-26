#!/usr/bin/env python3

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import numpy as np
import tensorflow as tf
import strawberryfields as sf
from strawberryfields.ops import *
import pickle
import random as rd
import warnings
import uncertainty_toolbox as uct

from dataloader import get_data, quartic_data
from plotting import plot_predictions



tf.get_logger().setLevel('ERROR')
warnings.filterwarnings("ignore")

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
rd.seed(42)


#TODO: try different functions
#TODO: store hyperparameter, variance, outputs etc. to show difference
#TODO: save outputs etc.  
#TODO: 010 pol , ause neg loglike as loss 


###### MLP BASELINE ######

#TODO: adapt hidden layer,  epochs, learning rate
def train_mlp_baseline(X_train, y_train, hidden_layers=[64, 64], epochs=100, learning_rate=0.01):
    """Train a simple MLP baseline model."""
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=(1,)))
    for units in hidden_layers:
        model.add(tf.keras.layers.Dense(units, activation='relu'))
    model.add(tf.keras.layers.Dense(1))

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='mse')

    model.fit(X_train, y_train, epochs=epochs, verbose=0)
    return model

def predict_mlp_baseline(model, X_test):
    """Predict using the trained MLP baseline model."""
    return model.predict(X_test).flatten()

###### POLYNOMIAL BASELINE ######

def train_polynomial_baseline(X_train, y_train, degree=3):
    """Fit a polynomial baseline model."""
    coeffs = np.polyfit(X_train.numpy(), y_train.numpy(), degree)
    return coeffs

def predict_polynomial_baseline(coeffs, X_test):
    """Predict using the fitted polynomial baseline model."""
    poly = np.poly1d(coeffs)
    return poly(X_test.numpy())


def selective_prediction(final_predictions, targets, predictive_uncertainty, threshold: float = 0.8):
    """ Copmutes UQ metrics
    
    Interpretation:
    - final_predictions: predictions from circuit, np.array
    - targets: data points, np.array
    - predictive_uncertainty: predictive uncertainty from ciruit, np.array 
    - threshold: threshold based on quantiles

    Returns:
     - final_predictions_sel: predictions from circuit, np.array
    - targets_sel: data points, np.array
    - predictive_uncertainty_sel: predictive uncertainty from ciruit, np.array
    - remaining_fraction: remaining data points after selective prediction
    """

    predictive_uncertainty_quantile = np.quantile(predictive_uncertainty, threshold)
    results_selected_mask = (predictive_uncertainty < predictive_uncertainty_quantile)

    # hopefully the mask works here
    final_predictions_sel = final_predictions[results_selected_mask]
    targets_sel = targets[results_selected_mask]
    predictive_uncertainty_sel = predictive_uncertainty[results_selected_mask]

    remaining_fraction = len(final_predictions_sel)/len(final_predictions)

    return final_predictions_sel, targets_sel, predictive_uncertainty_sel, remaining_fraction

def compute_eval_metrics(final_predictions, targets, predictive_uncertainty):
    #idea compute eval metrics for selective prediction and full version
    """ Copmutes UQ metrics
    
    Interpretation:
    - final_predictions: predictions from circuit, np.array
    - targets: data points, np.array
    - predictive_uncertainty: predictive uncertainty from ciruit, np.array 

    Returns:
        Dictionary containing all metrics. Accuracy metrics:  Mean average error ('mae'), Root mean squared
        error ('rmse'), Median absolute error ('mdae'),  Mean absolute
        relative percent difference ('marpd'), r^2 ('r2'), and Pearson's
        correlation coefficient ('corr').
    """
    if len(predictive_uncertainty) > 0:
    
        if len(final_predictions) > 0:
            uq_metrics = uct.metrics.get_all_metrics(
                final_predictions,
                predictive_uncertainty,
                targets,
                verbose=False,
                )
        else:
            uq_metrics = [] # TODO: define empty result
        # categories when predictive uncertainty is present
        uq_metric_categories = [
            "scoring_rule",
            "avg_calibration",
            "sharpness",
            "accuracy",
        ]

    else:
        # categories when no predictive uncertainty is present
        uq_metric_categories = ["accuracy"]
        uq_metrics = {
            "accuracy": uct.metrics.get_all_accuracy_metrics(
                final_predictions, targets
            )
        }

    return uq_metrics, uq_metric_categories

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

def train_memristor(x_train, y_train, memory_depth, training_steps):
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

            res_mem[('loss', 'tr', step)] = [loss.numpy(), phase1.numpy(), phase3.numpy(), memristor_weight.numpy()]
            print(f"Loss at step {step + 1}: {loss.numpy()}")

    print(f"Final loss: {loss.numpy()}")
    print(f"Optimal parameters: phase1={phase1.numpy()}, phase3={phase3.numpy()}, memristor_weight={memristor_weight.numpy()}")

    return res_mem, phase1, phase3, memristor_weight

def predict_memristor(x_test, y_test, memory_depth, phase1, phase3, memristor_weight, stochastic: bool = True, samples: int = 1, var: float = 0.1):
    """
    Uses the trained memristor model to make predictions on test data.
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
    targets = []

    # Initialize memory variables
    memory_p1 = tf.Variable(np.zeros(memory_depth), dtype=tf.float32)
    memory_p2 = tf.Variable(np.zeros(memory_depth), dtype=tf.float32)
    cycle_index = 0

    print("Predicting on test data...")

    for sample in range(samples):
        print(f"Sample {sample + 1}/{samples}")
        sample_predictions = []

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
            if sample == 0:
                targets.append(y_test[i])

        all_predictions.append(sample_predictions)

    # Convert all_predictions to a NumPy array for easier manipulation
    all_predictions = np.array(all_predictions)

    if stochastic:
        # Calculate mean and standard deviation along the column axis
        final_predictions = np.mean(all_predictions, axis=0)
        predictive_uncertainty = np.std(all_predictions, axis=0)
    else:
        final_predictions = all_predictions[0]
        predictive_uncertainty = np.zeros_like(final_predictions)


    return final_predictions, targets, predictive_uncertainty

def main():
    X_train, y_train, X_test, y_test, _ = get_data(n_data=100, sigma_noise_1=0.1, datafunction=quartic_data)

    # Train the memristor model
    res_mem, phase1, phase3, memristor_weight = train_memristor(X_train, y_train, memory_depth=3, training_steps=10)

    # Save training results
    with open("results_mem_t_lag_iris.pkl", "wb") as file:
        pickle.dump(res_mem, file)

    # Predict using the trained model
    predictions, targets, predictive_uncertainty = predict_memristor(
        X_test, y_test, memory_depth=3, phase1=phase1, phase3=phase3, memristor_weight=memristor_weight,
        stochastic=True, var=0.1, samples=3
    )

    # Ensure predictions and X_test have the same length
    assert len(predictions) == len(X_test), "Predictions and X_test must have the same length"

    # Convert predictions, targets, and predictive_uncertainty to NumPy arrays
    predictions = np.array(predictions)
    targets = np.array(targets)
    predictive_uncertainty = np.array(predictive_uncertainty)

    # Compute evaluation metrics for full predictions
    full_metrics, full_metric_categories = compute_eval_metrics(predictions, targets, predictive_uncertainty)
    print("Full Prediction Metrics:")
    for category in full_metric_categories:
        print(f"{category}: {full_metrics[category]}")

    # Apply selective prediction
    threshold = 0.8  # Example threshold
    sel_predictions, sel_targets, sel_uncertainty, remaining_fraction = selective_prediction(predictions, targets, predictive_uncertainty, threshold)
    print(f"Remaining Fraction after Selective Prediction: {remaining_fraction}")

    # Compute evaluation metrics for selective predictions
    sel_metrics, sel_metric_categories = compute_eval_metrics(sel_predictions, sel_targets, sel_uncertainty)
    print("Selective Prediction Metrics:")
    for category in sel_metric_categories:
        print(f"{category}: {sel_metrics[category]}")

    # Plotting the results
    plot_predictions(
        X_train.numpy(), y_train.numpy(), X_test.numpy(), y_test.numpy(),
        predictions, pred_std=predictive_uncertainty, epistemic=predictive_uncertainty,
        aleatoric=None, title="Memristor Model Predictions vs Targets"
    )

    # Train and predict with MLP baseline
    mlp_model = train_mlp_baseline(X_train, y_train)
    mlp_predictions = predict_mlp_baseline(mlp_model, X_test)
    plot_predictions(
        X_train.numpy(), y_train.numpy(), X_test.numpy(), y_test.numpy(),
        mlp_predictions, title="MLP Baseline Predictions"
    )

    # Train and predict with Polynomial baseline
    poly_coeffs = train_polynomial_baseline(X_train, y_train, degree=3)
    poly_predictions = predict_polynomial_baseline(poly_coeffs, X_test)
    plot_predictions(
        X_train.numpy(), y_train.numpy(), X_test.numpy(), y_test.numpy(),
        poly_predictions, title="Polynomial Baseline Predictions"
    )

    # # Plot all predictions together
    # plot_all_predictions(X_train, y_train, X_test, y_test, predictions, mlp_predictions, poly_predictions, predictive_uncertainty)

if __name__ == "__main__":
    main()


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