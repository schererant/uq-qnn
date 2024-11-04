#!/usr/bin/env python3

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import numpy as np
# Replace TensorFlow imports with PyTorch
# import tensorflow as tf
import torch
import torch.nn as nn
import torch.optim as optim
# import strawberryfields as sf
from strawberryfields.ops import *
import pickle
import random as rd
import matplotlib.pyplot as plt
import warnings

from collections.abc import Callable

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
rd.seed(42)

def quartic_data(input_data):
    """Create labels with quartic function."""
    y = input_data ** 4
    return y

def get_data(n_data: int =100, sigma_noise_1: float = 0.0, datafunction: Callable = quartic_data):
    """Define a function-based toy regression dataset."""
    x_min = 0
    x_max = 1
    X_train = torch.linspace(x_min, x_max, n_data)
    
    # Split training set
    gap_start = x_min + 0.35 * (x_max - x_min)
    gap_end = x_min + 0.6 * (x_max - x_min)

    # Create label noise
    noise_1 = torch.randn(n_data) * sigma_noise_1
    noise_1 = torch.where(X_train > gap_end, torch.tensor(0.0), noise_1)  # Only add noise to the left
    
    # Create labels
    y_train = datafunction(X_train)

    train_idx = (X_train < gap_end) & (X_train > gap_start)
    X_train = X_train[~train_idx]
    y_train = y_train[~train_idx]

    # Test data
    X_test = torch.linspace(x_min + x_min * 0.1, x_max - x_max * 0.1, 500)
    y_test = datafunction(X_test)

    return X_train, y_train, X_test, y_test, noise_1

# ...existing code...

def train_memristor(x_train, y_train, memory_depth):
    """
    Trains the memristor model using PyTorch.
    """
    # Initialize variables and optimizer
    phase1 = torch.rand(1, dtype=torch.float64, requires_grad=True) * 2 * np.pi
    phase3 = torch.rand(1, dtype=torch.float64, requires_grad=True) * 2 * np.pi
    memristor_weight = torch.rand(1, dtype=torch.float64, requires_grad=True)

    optimizer = optim.Adam([phase1, phase3, memristor_weight], lr=0.01)
    res_mem = {}

    encoded_phases = 2 * np.arccos(x_train.numpy())
    encoded_phases = torch.tensor(encoded_phases, dtype=torch.float64)
    num_samples = len(encoded_phases)

    # Initialize memory variables
    memory_p1 = torch.zeros(memory_depth, dtype=torch.float64)
    memory_p2 = torch.zeros(memory_depth, dtype=torch.float64)
    cycle_index = 0

    # Training loop
    for step in range(50):
        loss = 0.0
        optimizer.zero_grad()

        eng = sf.Engine(backend="fock", backend_options={"cutoff_dim": 4})
        for i in range(num_samples):
            time_step = i - cycle_index * memory_depth
            if time_step == memory_depth - 1:
                cycle_index += 1

            if i == 0:
                memristor_phase = torch.acos(torch.sqrt(torch.tensor(0.5)))
            else:
                memristor_phase = torch.acos(torch.sqrt(
                    memory_p1.sum() / memory_depth +
                    memristor_weight * memory_p2.sum() / memory_depth
                ))

            circuit = build_circuit(phase1, memristor_phase, phase3, encoded_phases[i])
            results = eng.run(circuit)

            # Get probabilities from the circuit results
            prob = results.state.all_fock_probs()
            prob = torch.tensor(prob, dtype=torch.float64)
            prob_state_001 = prob[0, 0, 1].real
            prob_state_010 = prob[0, 1, 0].real

            # Update memory variables
            index = time_step % memory_depth
            memory_p1[index] = prob_state_010
            memory_p2[index] = prob_state_001

            # Compute the loss
            loss += (y_train[i] - prob_state_001) ** 2

        # Backpropagation and optimization
        loss.backward()
        optimizer.step()

        res_mem[('loss', 'tr', step)] = [loss.item(), phase1.item(), phase3.item(), memristor_weight.item()]
        print(f"Loss at step {step + 1}: {loss.item()}")

    print(f"Final loss: {loss.item()}")
    print(f"Optimal parameters: phase1={phase1.item()}, phase3={phase3.item()}, memristor_weight={memristor_weight.item()}")

    return res_mem, phase1.detach(), phase3.detach(), memristor_weight.detach()

def predict_memristor(x_test, y_test, memory_depth, phase1, phase3, memristor_weight):
    """
    Uses the trained memristor model to make predictions on test data using PyTorch.
    """
    eng = sf.Engine(backend="fock", backend_options={"cutoff_dim": 4})
    encoded_phases = 2 * np.arccos(x_test.numpy())
    encoded_phases = torch.tensor(encoded_phases, dtype=torch.float64)

    predictions = []
    targets = []

    # Initialize memory variables
    memory_p1 = torch.zeros(memory_depth, dtype=torch.float64)
    memory_p2 = torch.zeros(memory_depth, dtype=torch.float64)
    cycle_index = 0

    for i in range(len(encoded_phases)):
        time_step = i - cycle_index * memory_depth
        if time_step == memory_depth - 1:
            cycle_index += 1

        if i == 0:
            memristor_phase = torch.acos(torch.sqrt(torch.tensor(0.5)))
        else:
            memristor_phase = torch.acos(torch.sqrt(
                memory_p1.sum() / memory_depth +
                memristor_weight * memory_p2.sum() / memory_depth
            ))

        circuit = build_circuit(phase1, memristor_phase, phase3, encoded_phases[i])
        results = eng.run(circuit)

        # Get probabilities from the circuit results
        prob = results.state.all_fock_probs()
        prob = torch.tensor(prob, dtype=torch.float64)
        prob_state_001 = prob[0, 0, 1].real
        prob_state_010 = prob[0, 1, 0].real

        # Update memory variables
        index = time_step % memory_depth
        memory_p1[index] = prob_state_010
        memory_p2[index] = prob_state_001

        # Store predictions and targets
        predictions.append(prob_state_001.item())
        targets.append(y_test[i].item())

    return predictions, targets

def main():

    X_train, y_train, X_test, y_test, _ = get_data(n_data=100, sigma_noise_1=0.0, datafunction=quartic_data)

    # Train the memristor model
    res_mem, phi1, phi3, x_2 = train_memristor(X_train, y_train, dip=3)

    # Save training results
    with open("results_mem_t_lag_iris.pkl", "wb") as file:
        pickle.dump(res_mem, file)

    # Predict using the trained model
    predictions, targets = predict_memristor(X_test, y_test, dip=3, phase1=phi1, phase3=phi3, memristor_weight=x_2)

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
