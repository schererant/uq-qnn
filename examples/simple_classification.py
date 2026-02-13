#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Simple binary classification example using the UQ-QNN framework.

This example demonstrates how to:
1. Generate synthetic classification data
2. Train a photonic neural network for classification
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
    """Run a simple binary classification example."""
    print("=== UQ-QNN: Simple Binary Classification Example ===")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Configure parameters
    config['n_data'] = 80
    config['lr'] = 0.03
    config['epochs'] = 30
    config['memory_depth'] = 2
    n_modes = 3
    n_phases = n_modes * (n_modes - 1)  # Clements: 3x3 = 6 phases
    config['phase_idx'] = tuple(range(n_phases))
    config['n_photons'] = tuple([1] * n_phases)
    n_samples = 500
    n_classes = 2  # Binary classification
    
    # Generate synthetic classification data
    print("Generating synthetic classification data...")
    X_train, y_train, X_test, y_test = get_classification_data(
        n_data=config['n_data'],
        n_classes=n_classes,
        data_type='binary_threshold',
        noise_level=0.05,
        return_one_hot=True  # 2D one-hot for cross-entropy loss
    )
    
    # Train the model with discrete phases
    print("Training model with discrete phases...")
    theta_discrete, history_discrete = train_pytorch(
        X_train, y_train,
        memory_depth=config['memory_depth'],
        lr=config['lr'],
        epochs=config['epochs'],
        n_swipe=0,
        n_samples=n_samples,
        n_modes=n_modes,
        memristive_phase_idx=[2],
        loss_type='cross_entropy',
        n_classes=n_classes,
        target_mode=(1, 2)  # Use modes 1 and 2 for binary classification
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
        memristive_phase_idx=[2],
        target_mode=(1, 2),
        return_class_probs=True
    )
    
    # Convert probabilities to class predictions
    # For binary: preds_probs[:, 1] is probability of class 1
    preds_discrete = np.argmax(preds_probs, axis=1)
    
    # Integer labels for evaluation (y_test is one-hot 2D)
    y_test_labels = np.argmax(y_test, axis=1)
    
    # Compute accuracy
    accuracy = accuracy_score(y_test_labels, preds_discrete)
    print(f"Test Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test_labels, preds_discrete))
    
    # Add uncertainty estimation through multiple forward passes
    print("\nEstimating uncertainty through multiple forward passes...")
    n_forward_passes = 10
    all_preds_probs = np.zeros((len(X_test), n_classes, n_forward_passes))
    
    for i in tqdm(range(n_forward_passes), desc="Forward passes"):
        # Each forward pass with a different sample count introduces some randomness
        sample_count = n_samples + np.random.randint(-100, 100)
        sample_count = max(100, sample_count)  # Ensure at least 100 samples
        
        # Small random perturbation to parameters to simulate quantum noise
        perturbed_theta = theta_discrete.copy()
        # Only perturb phases slightly, not the weight
        perturbed_theta[:-1] += np.random.normal(0, 0.05, size=len(perturbed_theta)-1)
        
        preds = run_simulation_sequence_np(
            perturbed_theta,
            config['memory_depth'],
            sample_count,
            encoded_phases=enc_test,
            n_modes=n_modes,
            memristive_phase_idx=[2],
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
    
    # Plot results
    plt.figure(figsize=(14, 10))
    
    # Plot 1: Training loss
    plt.subplot(2, 3, 1)
    plt.plot(history_discrete)
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss (Cross-Entropy)')
    plt.grid(True)
    
    # Plot 2: Data and predictions
    plt.subplot(2, 3, 2)
    colors = ['blue', 'red']
    for c in range(n_classes):
        mask = y_test_labels == c
        plt.scatter(X_test[mask], [c] * mask.sum(), c=colors[c], 
                   label=f'Class {c} (true)', alpha=0.6, s=20)
    pred_mask_0 = mean_preds == 0
    pred_mask_1 = mean_preds == 1
    plt.scatter(X_test[pred_mask_0], [0.1] * pred_mask_0.sum(), 
               c='cyan', marker='x', label='Predicted class 0', s=15)
    plt.scatter(X_test[pred_mask_1], [0.9] * pred_mask_1.sum(), 
               c='orange', marker='x', label='Predicted class 1', s=15)
    plt.xlabel('x')
    plt.ylabel('Class')
    plt.title('Classification Results')
    plt.legend()
    plt.grid(True)
    plt.ylim(-0.2, 1.2)
    
    # Plot 3: Class probabilities
    plt.subplot(2, 3, 3)
    plt.plot(X_test, mean_probs[:, 0], 'b-', label='P(Class 0)', alpha=0.7)
    plt.plot(X_test, mean_probs[:, 1], 'r-', label='P(Class 1)', alpha=0.7)
    plt.fill_between(X_test, 
                     mean_probs[:, 0] - std_probs[:, 0],
                     mean_probs[:, 0] + std_probs[:, 0],
                     color='blue', alpha=0.2)
    plt.fill_between(X_test,
                     mean_probs[:, 1] - std_probs[:, 1],
                     mean_probs[:, 1] + std_probs[:, 1],
                     color='red', alpha=0.2)
    plt.xlabel('x')
    plt.ylabel('Probability')
    plt.title('Class Probabilities')
    plt.legend()
    plt.grid(True)
    
    # Plot 4: Entropy (uncertainty)
    plt.subplot(2, 3, 4)
    scatter = plt.scatter(X_test, entropy, c=(mean_preds != y_test_labels), 
                         cmap='RdYlGn', alpha=0.7)
    plt.colorbar(scatter, label='Misclassified')
    plt.xlabel('x')
    plt.ylabel('Entropy (Uncertainty)')
    plt.title('Prediction Uncertainty')
    plt.grid(True)
    
    # Plot 5: Confusion matrix
    plt.subplot(2, 3, 5)
    cm = confusion_matrix(y_test_labels, mean_preds)
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
    
    # Plot 6: Calibration (entropy vs error)
    plt.subplot(2, 3, 6)
    errors = (mean_preds != y_test_labels).astype(int)
    plt.scatter(entropy, errors, alpha=0.7)
    plt.xlabel('Entropy (Uncertainty)')
    plt.ylabel('Error (0=correct, 1=wrong)')
    plt.title('Uncertainty vs. Error')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('classification_with_uncertainty.png', dpi=300)
    plt.show()
    
    # Print simulation statistics
    sim_logger.report()


if __name__ == "__main__":
    main()
