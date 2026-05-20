#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Two Moons (Half-Moons) Classification Example using the UQ-QNN framework.

Demonstrates 2D classification with Mauser-style multi-slot phase encoding,
L=3 re-uploading layers, decision-boundary visualisation, and optional
parameter-perturbation uncertainty (set unc_n_passes > 0).
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import get_two_moons_data, encode_2d_to_phases_multi
from src.experiment import Experiment

# ===================== EXPERIMENT CONFIG =====================
# Two moons needs more circuit capacity than the original n_modes=3/L=2 setup
# (only ~8 trainable phases + 2-dim readout → stuck at ~83% / loss 0.40).
# Wider mesh (n_modes=4) + 3 re-uploading layers gives 30 trainable phases and
# a 4-dim singles readout that the linear head can separate non-linearly.
N_LAYERS = 3
N_MODES = 4
PHASES_PER_LAYER = N_MODES * (N_MODES - 1)  # = 12 for n_modes=4
# Per Mauser re-uploading: x₁→slot 0, x₂→slot 1 of every layer block.
ENCODING_PHASE_IDX = tuple(
    layer * PHASES_PER_LAYER + s for layer in range(N_LAYERS) for s in (0, 1)
)
CONFIG = {
    # Circuit — Mauser-style: x₁→slot 0, x₂→slot 1 per layer; same x re-uploaded L times
    "n_modes": N_MODES,
    "n_layers": N_LAYERS,
    "input_state": (1, 0, 0, 0),
    "encoding_phase_idx": ENCODING_PHASE_IDX,
    "n_enc_features": 2,
    "photon_distinguishability": None,
    "target_mode": (1, 2),
    "memristive_phase_idx": None,
    "memristive_output_modes": None,
    # Task — binary classification
    "output_mode": "singles",
    "working_detectors": None,
    "loss_type": "cross_entropy",
    "n_classes": 2,
    # Training: deeper mesh has saturating regions; higher lr + more epochs to push through plateaus
    "lr": 0.04,
    "epochs": 1000,
    "n_samples": 2000,
    "memory_depth": 3,
    "n_swipe": 0,
    "swipe_span": 0.0,
    "noise_std": None,
    "seed": 0,
    # Backend
    "sim_backend": "numpy",
    # Data
    "data_n_samples": 2000,
    "data_noise": 0.08,
    # Uncertainty — 0 skips UQ; e.g. 5–20 enables parameter-perturbation analysis
    "unc_n_passes": 0,
    "unc_noise_std": 0.05,
}
# =============================================================


def _scatter_test(ax, X_test, y_test, n_classes, colors, *, s=15):
    for c in range(n_classes):
        mask = y_test == c
        ax.scatter(
            X_test[mask, 0],
            X_test[mask, 1],
            c=colors[c],
            alpha=0.6,
            s=s,
            edgecolors="black",
            linewidths=0.5,
        )


def main():
    n_classes = CONFIG["n_classes"]
    uq_enabled = CONFIG["unc_n_passes"] > 0
    eps = 1e-15

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

        enc_train = encode_2d_to_phases_multi(X_train, n_layers=N_LAYERS)
        enc_test = encode_2d_to_phases_multi(X_test, n_layers=N_LAYERS)

        trained_state, history, model = exp.train(enc_train, y_train, encoded=True)
        exp.save_metrics({"final_loss": history[-1], "uq_enabled": uq_enabled})

        preds_probs = model.predict(enc_test)
        preds_discrete = np.argmax(preds_probs, axis=1)

        accuracy = accuracy_score(y_test, preds_discrete)
        print(f"Test Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, preds_discrete))

        metrics: dict[str, float | bool] = {"test_accuracy": float(accuracy)}

        if uq_enabled:
            test_unc = exp.run_uncertainty_analysis(
                trained_state,
                enc_test,
                n_passes=CONFIG["unc_n_passes"],
                noise_std=CONFIG["unc_noise_std"],
            )
            if test_unc is not None:
                metrics["uq_mean_test_std"] = float(
                    test_unc["std"].max(axis=1).mean()
                )

        # Decision-boundary grid
        x_min, x_max = X_test[:, 0].min() - 0.1, X_test[:, 0].max() + 0.1
        y_min, y_max = X_test[:, 1].min() - 0.1, X_test[:, 1].max() + 0.1
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, 100),
            np.linspace(y_min, y_max, 100),
        )
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        enc_grid = encode_2d_to_phases_multi(grid_points, n_layers=N_LAYERS)
        grid_probs = model.predict(enc_grid)
        grid_preds = np.argmax(grid_probs, axis=1).reshape(xx.shape)
        grid_entropy = (-np.sum(grid_probs * np.log(grid_probs + eps), axis=1)).reshape(
            xx.shape
        )
        grid_p1 = grid_probs[:, 1].reshape(xx.shape)

        grid_std_p1 = None
        if uq_enabled:
            grid_unc = exp.run_uncertainty_analysis(
                trained_state,
                enc_grid,
                n_passes=CONFIG["unc_n_passes"],
                noise_std=CONFIG["unc_noise_std"],
            )
            if grid_unc is not None:
                grid_std_p1 = grid_unc["std"][:, 1].reshape(xx.shape)
                metrics["uq_mean_grid_std_p1"] = float(grid_std_p1.mean())

        exp.save_metrics(metrics)

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
                X_test[mask, 0],
                X_test[mask, 1],
                c=colors[c],
                label=f"Class {c}",
                alpha=0.6,
                s=20,
            )
        ax2.set(xlabel="x₁", ylabel="x₂", title="Test Data (True Labels)")
        ax2.legend()
        ax2.grid(True)

        ax3 = fig.add_subplot(2, 3, 3)
        ax3.contourf(xx, yy, grid_preds, alpha=0.3, levels=[0, 0.5, 1], colors=colors)
        _scatter_test(ax3, X_test, y_test, n_classes, colors, s=20)
        ax3.set(xlabel="x₁", ylabel="x₂", title="Decision Boundary")
        ax3.grid(True)

        ax4 = fig.add_subplot(2, 3, 4)
        cnt = ax4.contourf(xx, yy, grid_p1, levels=20, cmap="RdYlBu")
        fig.colorbar(cnt, ax=ax4, label="P(Class 1)")
        _scatter_test(ax4, X_test, y_test, n_classes, colors)
        ax4.set(xlabel="x₁", ylabel="x₂", title="Class 1 Probability")
        ax4.grid(True)

        ax5 = fig.add_subplot(2, 3, 5)
        if uq_enabled and grid_std_p1 is not None:
            cnt = ax5.contourf(xx, yy, grid_std_p1, levels=20, cmap="magma")
            fig.colorbar(cnt, ax=ax5, label="σ P(Class 1)")
            _scatter_test(ax5, X_test, y_test, n_classes, colors)
            ax5.set(xlabel="x₁", ylabel="x₂", title="Parameter Uncertainty (σ, grid)")
        else:
            cnt = ax5.contourf(xx, yy, grid_entropy, levels=20, cmap="viridis")
            fig.colorbar(cnt, ax=ax5, label="Entropy")
            _scatter_test(ax5, X_test, y_test, n_classes, colors)
            ax5.set(xlabel="x₁", ylabel="x₂", title="Model Confidence (Entropy)")
        ax5.grid(True)

        ax6 = fig.add_subplot(2, 3, 6)
        cm = confusion_matrix(y_test, preds_discrete)
        ax6.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        ax6.set(title="Confusion Matrix")
        ax6.set_xticks(range(n_classes), [f"Class {i}" for i in range(n_classes)])
        ax6.set_yticks(range(n_classes), [f"Class {i}" for i in range(n_classes)])
        thresh = cm.max() / 2.0
        for i, j in np.ndindex(cm.shape):
            ax6.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                color="white" if cm[i, j] > thresh else "black",
            )
        ax6.set(ylabel="True label", xlabel="Predicted label")

        fig.tight_layout()
        exp.savefig(fig, "two_moons_classification.png")
        plt.show()


if __name__ == "__main__":
    main()
