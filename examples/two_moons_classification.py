#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Two Moons (Half-Moons) Classification Example using the UQ-QNN framework.

Demonstrates 2D classification with custom phase encoding,
decision-boundary visualisation, and uncertainty estimation.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import get_two_moons_data, encode_2d_to_phase
from src.experiment import Experiment

# ===================== EXPERIMENT CONFIG =====================
CONFIG = {
    # Circuit
    "n_modes": 3,
    "input_state": (1, 0, 0),
    "encoding_phase_idx": 0,
    "photon_distinguishability": None,
    "target_mode": (1, 2),
    "memristive_phase_idx": [2],
    "memristive_output_modes": None,
    # Task — binary classification
    "output_mode": "singles",
    "working_detectors": None,
    "loss_type": "cross_entropy",
    "n_classes": 2,
    # Training
    "lr": 0.01,
    "epochs": 100,
    "n_samples": 1000,
    "memory_depth": 3,
    "n_swipe": 0,
    "swipe_span": 0.0,
    "noise_std": None,
    "seed": 42,
    # Backend
    "sim_backend": "numpy",
    # Data
    "data_n_samples": 1000,
    "data_noise": 0.05,
    "encoding_method": "radial",
    # Uncertainty
    "unc_n_passes": 10,
    "unc_noise_std": 0.05,
}
# =============================================================


def main():
    n_classes = CONFIG["n_classes"]

    with Experiment("Two Moons Classification", config=CONFIG) as exp:
        np.random.seed(CONFIG["seed"])

        X_train, y_train, X_test, y_test = get_two_moons_data(
            n_samples=CONFIG["data_n_samples"],
            noise=CONFIG["data_noise"],
            random_state=CONFIG["seed"],
            return_one_hot=False,
        )
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Test set:     {X_test.shape[0]} samples")

        enc_method = CONFIG["encoding_method"]
        enc_train = encode_2d_to_phase(X_train, method=enc_method)
        enc_test = encode_2d_to_phase(X_test, method=enc_method)

        theta, history = exp.train(enc_train, y_train, encoded=True)
        exp.save_metrics({"final_loss": history[-1]})

        preds_probs = exp.predict(theta, enc_test, return_class_probs=True)
        preds_discrete = np.argmax(preds_probs, axis=1)

        accuracy = accuracy_score(y_test, preds_discrete)
        print(f"Test Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, preds_discrete))

        unc = exp.run_uncertainty_analysis(
            theta, enc_test,
            n_passes=CONFIG["unc_n_passes"],
            noise_std=CONFIG["unc_noise_std"],
        )
        mean_probs = unc["mean"]
        mean_preds = np.argmax(mean_probs, axis=1)

        eps = 1e-15

        # Decision-boundary grid
        x_min, x_max = X_test[:, 0].min() - 0.1, X_test[:, 0].max() + 0.1
        y_min, y_max = X_test[:, 1].min() - 0.1, X_test[:, 1].max() + 0.1
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, 100),
            np.linspace(y_min, y_max, 100),
        )
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        enc_grid = encode_2d_to_phase(grid_points, method=enc_method)
        grid_probs = exp.predict(theta, enc_grid, return_class_probs=True)
        grid_preds = np.argmax(grid_probs, axis=1).reshape(xx.shape)
        grid_entropy = (
            -np.sum(grid_probs * np.log(grid_probs + eps), axis=1)
        ).reshape(xx.shape)
        grid_p1 = grid_probs[:, 1].reshape(xx.shape)

        exp.save_metrics({"test_accuracy": float(accuracy)})

        # ── plots ──
        fig = plt.figure(figsize=(18, 12))
        colors = ["blue", "red"]

        ax1 = fig.add_subplot(2, 3, 1)
        ax1.plot(history)
        ax1.set_yscale("log")
        ax1.set(xlabel="Epoch", ylabel="Loss", title="Training Loss (Cross-Entropy)")
        ax1.grid(True)

        ax2 = fig.add_subplot(2, 3, 2)
        for c in range(n_classes):
            mask = y_test == c
            ax2.scatter(
                X_test[mask, 0], X_test[mask, 1],
                c=colors[c], label=f"Class {c}", alpha=0.6, s=20,
            )
        ax2.set(xlabel="x₁", ylabel="x₂", title="Test Data (True Labels)")
        ax2.legend()
        ax2.grid(True)

        ax3 = fig.add_subplot(2, 3, 3)
        ax3.contourf(xx, yy, grid_preds, alpha=0.3, levels=[0, 0.5, 1], colors=colors)
        for c in range(n_classes):
            mask = y_test == c
            ax3.scatter(
                X_test[mask, 0], X_test[mask, 1],
                c=colors[c], label=f"Class {c}", alpha=0.6, s=20,
                edgecolors="black", linewidths=0.5,
            )
        ax3.set(xlabel="x₁", ylabel="x₂", title="Decision Boundary")
        ax3.legend()
        ax3.grid(True)

        ax4 = fig.add_subplot(2, 3, 4)
        cnt = ax4.contourf(xx, yy, grid_p1, levels=20, cmap="RdYlBu")
        fig.colorbar(cnt, ax=ax4, label="P(Class 1)")
        for c in range(n_classes):
            mask = y_test == c
            ax4.scatter(
                X_test[mask, 0], X_test[mask, 1],
                c=colors[c], alpha=0.6, s=15,
                edgecolors="black", linewidths=0.5,
            )
        ax4.set(xlabel="x₁", ylabel="x₂", title="Class 1 Probability")
        ax4.grid(True)

        ax5 = fig.add_subplot(2, 3, 5)
        cnt = ax5.contourf(xx, yy, grid_entropy, levels=20, cmap="viridis")
        fig.colorbar(cnt, ax=ax5, label="Entropy")
        for c in range(n_classes):
            mask = y_test == c
            ax5.scatter(
                X_test[mask, 0], X_test[mask, 1],
                c=colors[c], alpha=0.6, s=15,
                edgecolors="black", linewidths=0.5,
            )
        ax5.set(xlabel="x₁", ylabel="x₂", title="Prediction Uncertainty (Entropy)")
        ax5.grid(True)

        ax6 = fig.add_subplot(2, 3, 6)
        cm = confusion_matrix(y_test, mean_preds)
        ax6.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        ax6.set(title="Confusion Matrix")
        ax6.set_xticks(range(n_classes), [f"Class {i}" for i in range(n_classes)])
        ax6.set_yticks(range(n_classes), [f"Class {i}" for i in range(n_classes)])
        thresh = cm.max() / 2.0
        for i, j in np.ndindex(cm.shape):
            ax6.text(
                j, i, format(cm[i, j], "d"),
                ha="center", color="white" if cm[i, j] > thresh else "black",
            )
        ax6.set(ylabel="True label", xlabel="Predicted label")

        fig.tight_layout()
        exp.savefig(fig, "two_moons_classification.png")
        plt.show()


if __name__ == "__main__":
    main()
