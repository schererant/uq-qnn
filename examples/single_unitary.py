#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimal example: evaluate a single Clements circuit using PhotonicCircuit.

This replaces the older SimConfig + unitary_for_point pattern with the
refactored physics core so you can get singles / coincidences in a few lines.
"""

import os
import sys
from pprint import pprint

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import PhotonicCircuit


def main():
    # ------------------------------------------------------------------ setup
    n_modes = 6
    encoding_mode = 4  # which column receives the encoded photon

    rng = np.random.default_rng(42)
    phases = rng.uniform(0.0, 2 * np.pi, size=n_modes * (n_modes - 1))

    circuit = PhotonicCircuit(
        n_modes=n_modes,
        phases=phases,
        encoding_mode=encoding_mode,
    )
    print("Circuit config:")
    pprint(circuit.config)

    # ----------------------------------------------------------------- singles
    x = 0.5
    enc_phi = 2 * np.arccos(x)

    unitary = circuit.unitary(enc_phi)
    print("\nUnitary shape:", unitary.shape)
    print("Is unitary:", np.allclose(unitary @ unitary.conj().T, np.eye(n_modes)))

    singles = circuit.singles(enc_phi)
    print("\nSingles probabilities (sum to 1):")
    pprint(singles)

    target_prob = circuit.target_probability(
        enc_phi,
        target_modes=(1, 2, 3),
        average=False,
    )
    print("\nSelected output modes (1,2,3):", target_prob)

    # ------------------------------------------------------------- coincidences
    coincidences = circuit.coincidences(enc_phi, input_modes=(0, 1))
    print("\nCoincidence probabilities (first 5 channels):")
    pprint(coincidences[:5])

    # --------------------------------------------------------------- batch eval
    encoded_batch = np.linspace(0.0, np.pi, 8)
    singles_batch = circuit.singles_batch(encoded_batch)
    print("\nBatch singles array shape:", singles_batch.shape)


if __name__ == "__main__":
    main()
