import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.config import SimConfig
from src.numpy_backend import (
    clements_unitary,
    singles_prob_from_unitary,
    unitary_for_point,
)

# Build a SimConfig (only fields you need for a single forward pass)
cfg = SimConfig(
    n_modes=6,
    encoding_mode=4,
    target_mode=(1, 2, 3),
    memristive_phase_idx=None,
    memristive_output_modes=None,
    encoding_phase_idx=None,
    output_mode="singles",
    input_modes=None,
    working_detectors=None,
    noise_std=None,
    n_samples=20,
    memory_depth=2,
    n_photons=None,
    n_swipe=0,
    swipe_span=0.0,
    backend="numpy",
    loss_type="mse",
    n_classes=1,
)

# Random (or trained) phases — shape (n_modes * (n_modes-1),) = (30,) for n_modes=6
np.random.seed(42)
n_phases = cfg.n_modes * (cfg.n_modes - 1)
phases = np.random.uniform(0, 2 * np.pi, n_phases)

# Encode a single input value (e.g. x=0.5 → phase)
x = 0.5
enc_phi = 2 * np.arccos(x)

# Compute the full unitary for this one data point
U = unitary_for_point(
    phases, enc_phi, cfg.n_modes, cfg.encoding_mode, cfg.encoding_phase_idx
)

print("Unitary shape:", U.shape)  # (6, 6)
print("Is unitary:", np.allclose(U @ U.conj().T, np.eye(cfg.n_modes)))

# Extract the output probability (Born rule)
prob = singles_prob_from_unitary(U, cfg.encoding_mode, cfg.target_mode)
print("Output probability:", prob)
