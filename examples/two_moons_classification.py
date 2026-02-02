#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Two Moons (Half-Moons) Classification Example using the UQ-QNN framework.

This example demonstrates how to:
1. Generate the classic Two Moons 2D classification dataset
2. Encode 2D features for photonic circuit input
3. Train a photonic neural network for binary classification
4. Visualize the decision boundary and uncertainty
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Add the parent directory to the path so we can import the library
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import get_two_moons_data, encode_2d_to_phase
from src.training import train_pytorch_generic
from src.simulation import run_simulation_sequence_np, sim_logger
from src.utils import config


def main():
    """Run a Two Moons classification example."""
    print("=== UQ-QNN: Two Moons Classification Example ===")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Configure parameters
    config['lr'] = 0.03
    config['epochs'] = 50
    config['memory_depth'] = 2
    config['phase_idx'] = (0, 1)  # Indices of phase parameters (excluding weight)
    config['n_photons'] = (1, 1)  # Number of photons for each phase
    n_samples = 500
    n_phases = 2
    n_classes = 2  # Binary classification
    
    # Generate Two Moons dataset
    print("Generating Two Moons dataset...")
    X_train, y_train, X_test, y_test = get_two_moons_data(
        n_samples=1000,
        noise=0.1,
        random_state=42,
        return_one_hot=False
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Input dimension: {X_train.shape[1]}D")
    
    # Encode 2D features to phase values
    print("Encoding 2D features to phase values...")
    enc_train = encode_2d_to_phase(X_train, method='weighted_sum')
    enc_test = encode_2d_to_phase(X_test, method='weighted_sum')
    
    # Train the model with discrete phases
    print("Training model with discrete phases...")
    theta_discrete, history_discrete = train_pytorch_generic(
        enc_train, y_train,
        memory_depth=config['memory_depth'],
        lr=config['lr'],
        epochs=config['epochs'],
        phase_idx=config['phase_idx'],
        n_photons=config['n_photons'],
        n_swipe=0,
        n_samples=n_samples,
        n_phases=n_phases,
        loss_type='cross_entropy',
        n_classes=n_classes,
        target_mode=(1, 2)  # Use modes 1 and 2 for binary classification
    )
    
    # Generate predictions
    print("Generating predictions...")
    preds_probs = run_simulation_sequence_np(
        theta_discrete, 
        config['memory_depth'], 
        n_samples, 
        encoded_phases=enc_test,
        target_mode=(1, 2),
        return_class_probs=True
    )
    
    # Convert probabilities to class predictions
    preds_discrete = np.argmax(preds_probs, axis=1)
    
    # Compute accuracy
    accuracy = accuracy_score(y_test, preds_discrete)
    print(f"Test Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, preds_discrete))
    
    # Add uncertainty estimation through multiple forward passes
    print("\nEstimating uncertainty through multiple forward passes...")
    n_forward_passes = 10
    all_preds_probs = np.zeros((len(X_test), n_classes, n_forward_passes))
    
    for i in tqdm(range(n_forward_passes), desc="Forward passes"):
        # Each forward pass with a different sample count introduces some randomness
        sample_count = n_samples + np.random.randint(-100, 100)
        sample_count = max(100, sample_count)
        
        # Small random perturbation to parameters to simulate quantum noise
        perturbed_theta = theta_discrete.copy()
        perturbed_theta[:-1] += np.random.normal(0, 0.05, size=len(perturbed_theta)-1)
        
        preds = run_simulation_sequence_np(
            perturbed_theta, 
            config['memory_depth'], 
            sample_count, 
            encoded_phases=enc_test,
            target_mode=(1, 2),
            return_class_probs=True
        )
        all_preds_probs[:, :, i] = preds
    
    # Compute mean probabilities and entropy (uncertainty measure)
    mean_probs = np.mean(all_preds_probs, axis=2)
    std_probs = np.std(all_preds_probs, axis=2)
    
    # Entropy as uncertainty measure: H = -Σ p_c * log(p_c)
    eps = 1e-15
    entropy = -np.sum(mean_probs * np.log(mean_probs + eps), axis=1)
    
    # Final predictions from mean probabilities
    mean_preds = np.argmax(mean_probs, axis=1)
    
    # Create a grid for decision boundary visualization
    print("Creating decision boundary visualization...")
    x_min, x_max = X_test[:, 0].min() - 0.1, X_test[:, 0].max() + 0.1
    y_min, y_max = X_test[:, 1].min() - 0.1, X_test[:, 1].max() + 0.1
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 100),
        np.linspace(y_min, y_max, 100)
    )
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    
    # Encode grid points and get predictions
    enc_grid = encode_2d_to_phase(grid_points, method='weighted_sum')
    grid_probs = run_simulation_sequence_np(
        theta_discrete,
        config['memory_depth'],
        n_samples,
        encoded_phases=enc_grid,
        target_mode=(1, 2),
        return_class_probs=True
    )
    grid_preds = np.argmax(grid_probs, axis=1)
    grid_entropy = -np.sum(grid_probs * np.log(grid_probs + eps), axis=1)
    
    # Reshape for plotting
    Z_pred = grid_preds.reshape(xx.shape)
    Z_entropy = grid_entropy.reshape(xx.shape)
    Z_prob_class1 = grid_probs[:, 1].reshape(xx.shape)
    
    # Plot results
    fig = plt.figure(figsize=(18, 12))
    
    # Plot 1: Training loss
    plt.subplot(2, 3, 1)
    plt.plot(history_discrete)
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss (Cross-Entropy)')
    plt.grid(True)
    
    # Plot 2: Original data with true labels
    plt.subplot(2, 3, 2)
    colors = ['blue', 'red']
    for c in range(n_classes):
        mask = y_test == c
        plt.scatter(X_test[mask, 0], X_test[mask, 1], c=colors[c], 
                   label=f'Class {c}', alpha=0.6, s=20)
    plt.xlabel('x₁')
    plt.ylabel('x₂')
    plt.title('Test Data (True Labels)')
    plt.legend()
    plt.grid(True)
    
    # Plot 3: Decision boundary
    plt.subplot(2, 3, 3)
    plt.contourf(xx, yy, Z_pred, alpha=0.3, levels=[0, 0.5, 1], colors=colors)
    for c in range(n_classes):
        mask = y_test == c
        plt.scatter(X_test[mask, 0], X_test[mask, 1], c=colors[c], 
                   label=f'Class {c}', alpha=0.6, s=20, edgecolors='black', linewidths=0.5)
    plt.xlabel('x₁')
    plt.ylabel('x₂')
    plt.title('Decision Boundary')
    plt.legend()
    plt.grid(True)
    
    # Plot 4: Class probability surface (Class 1)
    plt.subplot(2, 3, 4)
    contour = plt.contourf(xx, yy, Z_prob_class1, levels=20, cmap='RdYlBu')
    plt.colorbar(contour, label='P(Class 1)')
    for c in range(n_classes):
        mask = y_test == c
        plt.scatter(X_test[mask, 0], X_test[mask, 1], c=colors[c], 
                   alpha=0.6, s=15, edgecolors='black', linewidths=0.5)
    plt.xlabel('x₁')
    plt.ylabel('x₂')
    plt.title('Class 1 Probability')
    plt.grid(True)
    
    # Plot 5: Uncertainty (Entropy) surface
    plt.subplot(2, 3, 5)
    contour = plt.contourf(xx, yy, Z_entropy, levels=20, cmap='viridis')
    plt.colorbar(contour, label='Entropy')
    for c in range(n_classes):
        mask = y_test == c
        plt.scatter(X_test[mask, 0], X_test[mask, 1], c=colors[c], 
                   alpha=0.6, s=15, edgecolors='black', linewidths=0.5)
    plt.xlabel('x₁')
    plt.ylabel('x₂')
    plt.title('Prediction Uncertainty (Entropy)')
    plt.grid(True)
    
    # Plot 6: Confusion matrix
    plt.subplot(2, 3, 6)
    cm = confusion_matrix(y_test, mean_preds)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(n_classes)
    plt.xticks(tick_marks, [f'Class {i}' for i in range(n_classes)])
    plt.yticks(tick_marks, [f'Class {i}' for i in range(n_classes)])
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    plt.tight_layout()
    plt.savefig('two_moons_classification.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print simulation statistics
    sim_logger.report()


if __name__ == "__main__":
    main()
