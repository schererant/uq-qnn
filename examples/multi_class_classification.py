#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Multi-class (3-class) classification example using the UQ-QNN framework."""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import get_classification_data
from src.experiment import Experiment

# ===================== EXPERIMENT CONFIG =====================
CONFIG = {
    # Circuit
    "n_modes": 4,
    "encoding_mode": 0,
    "target_mode": (0, 1, 2),
    "memristive_phase_idx": None,
    "memristive_output_modes": None,
    "encoding_phase_idx": None,
    # Task — 3-class classification
    "output_mode": "singles",
    "input_modes": None,
    "working_detectors": None,
    "loss_type": "cross_entropy",
    "n_classes": 3,
    # Training
    "lr": 0.03,
    "epochs": 40,
    "n_samples": 500,
    "memory_depth": 2,
    "n_photons": None,
    "n_swipe": 0,
    "swipe_span": 0.0,
    "noise_std": None,
    "seed": 42,
    # Backend
    "sim_backend": "numpy",
    # Data
    "n_data": 100,
    "data_noise": 0.05,
    # Uncertainty
    "unc_n_passes": 10,
    "unc_noise_std": 0.05,
}
# =============================================================


def main():
    n_classes = CONFIG["n_classes"]

    with Experiment("Multi-Class Classification", config=CONFIG) as exp:
        np.random.seed(CONFIG["seed"])

        X_train, y_train, X_test, y_test = get_classification_data(
            n_data=CONFIG["n_data"],
            n_classes=n_classes,
            data_type="multi_class_regions",
            noise_level=CONFIG["data_noise"],
            return_one_hot=False,
        )

        theta, history = exp.train(X_train, y_train)
        exp.save_metrics({"final_loss": history[-1]})

        enc_test = 2 * np.arccos(X_test)

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
        std_probs = unc["std"]
        mean_preds = np.argmax(mean_probs, axis=1)

        eps = 1e-15
        entropy = -np.sum(mean_probs * np.log(mean_probs + eps), axis=1)

        exp.save_metrics({"test_accuracy": float(accuracy)})

        # ── plots ──
        colors = ["blue", "green", "red"]
        fig, axes_flat = plt.subplots(2, 4, figsize=(16, 10))

        axes_flat[0, 0].plot(history)
        axes_flat[0, 0].set_yscale("log")
        axes_flat[0, 0].set(xlabel="Epoch", ylabel="Loss", title="Training Loss")
        axes_flat[0, 0].grid(True)

        ax_res = axes_flat[0, 1]
        for c in range(n_classes):
            mask = y_test == c
            ax_res.scatter(
                X_test[mask], [c] * mask.sum(),
                c=colors[c], label=f"Class {c} (true)", alpha=0.6, s=20,
            )
        for c in range(n_classes):
            pred_mask = mean_preds == c
            if pred_mask.sum() > 0:
                ax_res.scatter(
                    X_test[pred_mask], [c + 0.1] * pred_mask.sum(),
                    c=colors[c], marker="x", label=f"Pred {c}", s=15,
                )
        ax_res.set(xlabel="x", ylabel="Class", title="Classification Results")
        ax_res.legend(fontsize=7)
        ax_res.set_ylim(-0.2, n_classes + 0.2)
        ax_res.grid(True)

        for c in range(n_classes):
            ax = axes_flat[0, 2] if c == 0 else axes_flat[0, 3] if c == 1 else axes_flat[1, 0]
            ax.plot(
                X_test, mean_probs[:, c],
                color=colors[c], label=f"P(Class {c})", linewidth=2, alpha=0.7,
            )
            ax.fill_between(
                X_test,
                mean_probs[:, c] - std_probs[:, c],
                mean_probs[:, c] + std_probs[:, c],
                color=colors[c], alpha=0.2,
            )
            ax.set(xlabel="x", ylabel="Probability", title=f"Class {c} Probability")
            ax.legend()
            ax.set_ylim(-0.1, 1.1)
            ax.grid(True)

        sc = axes_flat[1, 1].scatter(
            X_test, entropy, c=(mean_preds != y_test), cmap="RdYlGn", alpha=0.7
        )
        fig.colorbar(sc, ax=axes_flat[1, 1], label="Misclassified")
        axes_flat[1, 1].set(xlabel="x", ylabel="Entropy", title="Prediction Uncertainty")
        axes_flat[1, 1].grid(True)

        ax_cm = axes_flat[1, 2]
        cm = confusion_matrix(y_test, mean_preds)
        ax_cm.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        ax_cm.set(title="Confusion Matrix")
        ax_cm.set_xticks(range(n_classes), [f"C{i}" for i in range(n_classes)])
        ax_cm.set_yticks(range(n_classes), [f"C{i}" for i in range(n_classes)])
        thresh = cm.max() / 2.0
        for i, j in np.ndindex(cm.shape):
            ax_cm.text(
                j, i, format(cm[i, j], "d"),
                ha="center", color="white" if cm[i, j] > thresh else "black",
            )
        ax_cm.set(ylabel="True label", xlabel="Predicted label")

        axes_flat[1, 3].scatter(
            entropy, (mean_preds != y_test).astype(int), alpha=0.7
        )
        axes_flat[1, 3].set(
            xlabel="Entropy", ylabel="Error (0=correct, 1=wrong)",
            title="Uncertainty vs. Error",
        )
        axes_flat[1, 3].grid(True)

        fig.tight_layout()
        exp.savefig(fig, "multi_class_classification.png")
        plt.show()


if __name__ == "__main__":
    main()
