#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimal example: evaluate Clements circuits using PhotonicCircuit.

Uses internal mesh encoding at ``encoding_phase_idx`` and an explicit
``input_state`` for singles vs two-photon coincidence demos.
"""

import os
import sys
from pprint import pprint

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import PhotonicCircuit
from src.config import CircuitConfig


def main():
    # ------------------------------------------------------------------ setup
    n_modes = 6
    singles_cc = CircuitConfig(
        n_modes=n_modes,
        input_state=(4,),
        encoding_phase_idx=0,
        photon_distinguishability=None,
        output_mode="singles",
        working_detectors=None,
    )
    coinc_cc = CircuitConfig(
        n_modes=n_modes,
        input_state=(0, 1),
        encoding_phase_idx=0,
        photon_distinguishability="indistinguishable",
        output_mode="coincidence",
        working_detectors=tuple(range(n_modes)),
    )

    rng = np.random.default_rng(42)
    phases = rng.uniform(0.0, 2 * np.pi, size=n_modes * (n_modes - 1))

    circuit_s = PhotonicCircuit(
        n_modes=n_modes,
        phases=phases,
        circuit_config=singles_cc,
    )
    print("Singles circuit config:")
    pprint(circuit_s.config)

    # ----------------------------------------------------------------- singles
    x = 0.5
    enc_phi = 2 * np.arccos(x)

    unitary = circuit_s.unitary(enc_phi)
    print("\nUnitary shape:", unitary.shape)
    print("Is unitary:", np.allclose(unitary @ unitary.conj().T, np.eye(n_modes)))

    singles = circuit_s.singles(enc_phi)
    print("\nSingles probabilities (sum to 1):")
    pprint(singles)

    target_prob = circuit_s.target_probability(
        enc_phi,
        target_modes=(1, 2, 3),
        average=False,
    )
    print("\nSelected output modes (1,2,3):", target_prob)

    # ------------------------------------------------------------- coincidences
    circuit_c = PhotonicCircuit(
        n_modes=n_modes,
        phases=phases,
        circuit_config=coinc_cc,
    )
    coincidences = circuit_c.coincidences(enc_phi)
    print("\nCoincidence probabilities (first 5 channels):")
    pprint(coincidences[:5])

    # --------------------------------------------------------------- batch eval
    encoded_batch = np.linspace(0.0, np.pi, 8)
    singles_batch = circuit_s.singles_batch(encoded_batch)
    print("\nBatch singles array shape:", singles_batch.shape)


if __name__ == "__main__":
    main()
