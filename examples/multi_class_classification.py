#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Multi-class classification example using the UQ-QNN framework.

This example demonstrates how to:
1. Generate synthetic multi-class classification data
2. Train a photonic neural network for multi-class classification
3. Evaluate predictions, accuracy, and uncertainty
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Add the parent directory to the path so we can import the library
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import get_classification_data
from src.training import train_pytorch
from src.simulation import run_simulation_sequence_np, sim_logger
from src.utils import config


def main():
    """Run a multi-class classification example."""
    print("=== UQ-QNN: Multi-Class Classification Example ===")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Configure parameters
    config['n_data'] = 100
    config['lr'] = 0.03
    config['epochs'] = 40
    config['memory_depth'] = 2
    config['phase_idx'] = (0, 1)  # Indices of phase parameters (excluding weight)
    config['n_photons'] = (1, 1)  # Number of photons for each phase
    n_samples = 500
    n_phases = 2
    n_classes = 3  # Three-class classification
    
    n_modes = 4  # Need at least n_classes modes (Clements architecture)
    
    # Generate synthetic multi-class classification data
    print("Generating synthetic multi-class classification data...")
    X_train, y_train, X_test, y_test = get_classification_data(
        n_data=config['n_data'],
        n_classes=n_classes,
        data_type='multi_class_regions',
        noise_level=0.05,
        return_one_hot=False
    )
    
    # Train the model with discrete phases
    print("Training model with discrete phases...")
    theta_discrete, history_discrete = train_pytorch(
        X_train, y_train,
        memory_depth=config['memory_depth'],
        lr=config['lr'],
        epochs=config['epochs'],
        phase_idx=config['phase_idx'],
        n_photons=config['n_photons'],
        n_swipe=0,
        n_samples=n_samples,
        n_phases=n_phases,
        n_modes=n_modes,
        loss_type='cross_entropy',
        n_classes=n_classes,
        target_mode=(0, 1, 2)  # Use first 3 modes for 3 classes
    )
    
    # Generate predictions
    print("Generating predictions...")
    enc_test = 2 * np.arccos(X_test)
    preds_probs = run_simulation_sequence_np(
        theta_discrete, 
        config['memory_depth'], 
        n_samples, 
        encoded_phases=enc_test,
        n_modes=n_modes,
        target_mode=(0, 1, 2),
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
        
        # Small random perturbation to parameters
        perturbed_theta = theta_discrete.copy()
        perturbed_theta[:-1] += np.random.normal(0, 0.05, size=len(perturbed_theta)-1)
        # Ensure phases stay in [0, 2π)
        perturbed_theta[:-1] = perturbed_theta[:-1] % (2 * np.pi)
        
        preds = run_simulation_sequence_np(
            perturbed_theta, 
            config['memory_depth'], 
            sample_count, 
            encoded_phases=enc_test,
            n_modes=n_modes,
            target_mode=(0, 1, 2),
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
    
    # Plot results
    plt.figure(figsize=(16, 10))
    
    # Plot 1: Training loss
    plt.subplot(2, 4, 1)
    plt.plot(history_discrete)
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss (Cross-Entropy)')
    plt.grid(True)
    
    # Plot 2: Data and predictions
    plt.subplot(2, 4, 2)
    colors = ['blue', 'green', 'red']
    for c in range(n_classes):
        mask = y_test == c
        plt.scatter(X_test[mask], [c] * mask.sum(), c=colors[c], 
                   label=f'Class {c} (true)', alpha=0.6, s=20)
    for c in range(n_classes):
        pred_mask = mean_preds == c
        if pred_mask.sum() > 0:
            plt.scatter(X_test[pred_mask], [c + 0.1] * pred_mask.sum(), 
                       c=colors[c], marker='x', label=f'Pred class {c}', s=15)
    plt.xlabel('x')
    plt.ylabel('Class')
    plt.title('Classification Results')
    plt.legend()
    plt.grid(True)
    plt.ylim(-0.2, n_classes + 0.2)
    
    # Plot 3-5: Class probabilities for each class
    for c in range(n_classes):
        plt.subplot(2, 4, 3 + c)
        plt.plot(X_test, mean_probs[:, c], color=colors[c], 
                label=f'P(Class {c})', alpha=0.7, linewidth=2)
        plt.fill_between(X_test,
                         mean_probs[:, c] - std_probs[:, c],
                         mean_probs[:, c] + std_probs[:, c],
                         color=colors[c], alpha=0.2)
        plt.xlabel('x')
        plt.ylabel('Probability')
        plt.title(f'Class {c} Probability')
        plt.legend()
        plt.grid(True)
        plt.ylim(-0.1, 1.1)
    
    # Plot 6: Entropy (uncertainty)
    plt.subplot(2, 4, 6)
    scatter = plt.scatter(X_test, entropy, c=(mean_preds != y_test), 
                         cmap='RdYlGn', alpha=0.7)
    plt.colorbar(scatter, label='Misclassified')
    plt.xlabel('x')
    plt.ylabel('Entropy (Uncertainty)')
    plt.title('Prediction Uncertainty')
    plt.grid(True)
    
    # Plot 7: Confusion matrix
    plt.subplot(2, 4, 7)
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
    
    # Plot 8: Calibration (entropy vs error)
    plt.subplot(2, 4, 8)
    errors = (mean_preds != y_test).astype(int)
    plt.scatter(entropy, errors, alpha=0.7)
    plt.xlabel('Entropy (Uncertainty)')
    plt.ylabel('Error (0=correct, 1=wrong)')
    plt.title('Uncertainty vs. Error')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('multi_class_classification.png', dpi=300)
    plt.show()
    
    # Print simulation statistics
    sim_logger.report()


if __name__ == "__main__":
    main()
