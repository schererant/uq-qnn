#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Simple binary classification example using the UQ-QNN framework."""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import get_classification_data
from src.experiment import Experiment

# ===================== EXPERIMENT CONFIG =====================
CONFIG = {
    # Circuit
    "n_modes": 3,
    "input_state": (1, 0, 0),
    "encoding_phase_idx": 1,
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
    "lr": 0.03,
    "epochs": 10,
    "n_samples": 500,
    "memory_depth": 2,
    "n_swipe": 0,
    "swipe_span": 0.0,
    "noise_std": None,
    "seed": 42,
    # Backend
    "sim_backend": "numpy",
    # Data
    "n_data": 80,
    "data_noise": 0.05,
    # Uncertainty
    "unc_n_passes": 10,
    "unc_noise_std": 0.05,
}
# =============================================================


def main():
    n_classes = CONFIG["n_classes"]

    with Experiment("Simple Classification", config=CONFIG) as exp:
        np.random.seed(CONFIG["seed"])

        X_train, y_train, X_test, y_test = get_classification_data(
            n_data=CONFIG["n_data"],
            n_classes=n_classes,
            data_type="binary_threshold",
            noise_level=CONFIG["data_noise"],
            return_one_hot=True,
        )

        theta, history, model = exp.train(X_train, y_train)
        exp.save_metrics({"final_loss": history[-1]})

        enc_test = 2 * np.arccos(X_test)
        unc = exp.run_uncertainty_analysis(
            theta,
            enc_test,
            n_passes=CONFIG["unc_n_passes"],
            noise_std=CONFIG["unc_noise_std"],
            model=model,
        )
        mean_probs = unc["mean"]
        std_probs = unc["std"]
        mean_preds = np.argmax(mean_probs, axis=1)

        eps = 1e-15
        entropy = -np.sum(mean_probs * np.log(mean_probs + eps), axis=1)

        y_test_labels = np.argmax(y_test, axis=1)
        accuracy = accuracy_score(y_test_labels, mean_preds)
        print(f"Test Accuracy: {accuracy:.4f}")
        exp.save_metrics({"test_accuracy": accuracy})

        # ── plots ──
        fig, axes = plt.subplots(2, 3, figsize=(14, 10))

        axes[0, 0].plot(history)
        axes[0, 0].set_yscale("log")
        axes[0, 0].set(xlabel="Epoch", ylabel="Loss", title="Training Loss")
        axes[0, 0].grid(True)

        colors = ["blue", "red"]
        for c in range(n_classes):
            mask = y_test_labels == c
            axes[0, 1].scatter(
                X_test[mask],
                [c] * mask.sum(),
                c=colors[c],
                label=f"Class {c}",
                alpha=0.6,
            )
        axes[0, 1].scatter(
            X_test, mean_preds + 0.1, c="orange", marker="x", label="Preds", s=15
        )
        axes[0, 1].set(xlabel="x", ylabel="Class", title="Classification Results")
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        axes[0, 2].plot(X_test, mean_probs[:, 0], "b-", label="P(C0)")
        axes[0, 2].plot(X_test, mean_probs[:, 1], "r-", label="P(C1)")
        axes[0, 2].fill_between(
            X_test,
            mean_probs[:, 0] - std_probs[:, 0],
            mean_probs[:, 0] + std_probs[:, 0],
            color="blue",
            alpha=0.1,
        )
        axes[0, 2].fill_between(
            X_test,
            mean_probs[:, 1] - std_probs[:, 1],
            mean_probs[:, 1] + std_probs[:, 1],
            color="red",
            alpha=0.1,
        )
        axes[0, 2].set(title="Class Probabilities")
        axes[0, 2].legend()
        axes[0, 2].grid(True)

        axes[1, 0].scatter(
            X_test, entropy, c=(mean_preds != y_test_labels), cmap="RdYlGn"
        )
        axes[1, 0].set(title="Uncertainty (Entropy)")
        axes[1, 0].grid(True)

        cm = confusion_matrix(y_test_labels, mean_preds)
        axes[1, 1].imshow(cm, cmap=plt.cm.Blues)
        axes[1, 1].set(title="Confusion Matrix")
        for i, j in np.ndindex(cm.shape):
            axes[1, 1].text(j, i, cm[i, j], ha="center", va="center")

        axes[1, 2].scatter(
            entropy, (mean_preds != y_test_labels).astype(int), alpha=0.5
        )
        axes[1, 2].set(xlabel="Entropy", ylabel="Error", title="Uncertainty vs. Error")
        axes[1, 2].grid(True)

        fig.tight_layout()
        exp.savefig(fig, "classification_with_uncertainty.png")
        # plt.show()


if __name__ == "__main__":
    main()
