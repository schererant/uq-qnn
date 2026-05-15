"""Tests for Mauser-style multi-slot / multi-layer encoding."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.autograd import MemristorLossPSR
from src.circuits import build_circuit
from src.config import SimConfig, validate_sim_config
from src.data import encode_2d_to_phases_multi
from src.numpy_backend import clements_stack_unitary_batch
from src.simulation import run_simulation_sequence_np


def _two_moons_cfg(**kwargs: object) -> SimConfig:
    base = dict(
        n_modes=3,
        n_layers=2,
        input_state=(1, 0, 0),
        encoding_phase_idx=(0, 1, 6, 7),
        n_enc_features=2,
        photon_distinguishability=None,
        target_mode=(1, 2),
        memristive_phase_idx=None,
        memristive_output_modes=None,
        output_mode="singles",
        working_detectors=None,
        noise_std=None,
        n_samples=50,
        memory_depth=1,
        n_swipe=0,
        swipe_span=0.0,
        backend="numpy",
        loss_type="cross_entropy",
        n_classes=2,
        feedback_mode="internal_arm",
        feedback_modes=None,
        computation_modes=None,
    )
    base.update(kwargs)
    return SimConfig(**base)


def test_stacked_unitary_is_unitary() -> None:
    n_modes = 3
    n_layers = 2
    n_ph = n_modes * (n_modes - 1)
    rng = np.random.default_rng(0)
    phases = rng.uniform(0, 2 * np.pi, size=n_layers * n_ph)
    enc = rng.uniform(0, np.pi, size=(3, 4))
    slots = (0, 1, 6, 7)
    u = clements_stack_unitary_batch(phases, enc, slots, n_modes, n_layers)
    for i in range(u.shape[0]):
        uu = u[i] @ u[i].conj().T
        np.testing.assert_allclose(uu, np.eye(n_modes), atol=1e-10)


def test_encode_2d_to_phases_multi_shape() -> None:
    x = np.array([[0.2, 0.8], [0.5, 0.5]])
    enc = encode_2d_to_phases_multi(x, n_layers=2)
    assert enc.shape == (2, 4)
    np.testing.assert_allclose(enc[:, 0], enc[:, 2])
    np.testing.assert_allclose(enc[:, 1], enc[:, 3])


def test_multi_layer_numpy_perceval_agreement() -> None:
    pcvl = pytest.importorskip("perceval")
    from perceval.algorithm import Sampler

    cfg = _two_moons_cfg(backend="perceval")
    validate_sim_config(cfg)
    n_ph = cfg.total_mesh_phases
    rng = np.random.default_rng(1)
    theta = rng.uniform(0, 2 * np.pi, size=n_ph)
    x = np.array([[0.3, 0.7], [0.6, 0.4]])
    enc = encode_2d_to_phases_multi(x, n_layers=2)

    np_preds = run_simulation_sequence_np(
        theta, enc, cfg, return_class_probs=True
    )

    pv_preds = []
    for row in enc:
        circ = build_circuit(
            theta,
            row,
            n_modes=3,
            encoding_phase_idx=(0, 1, 6, 7),
            n_layers=2,
        )
        proc = pcvl.Processor("SLOS", circ)
        proc.with_input(pcvl.BasicState([1, 0, 0]))
        pv = Sampler(proc).probs(500)["results"]
        pv_preds.append(
            [
                pv.get(pcvl.BasicState([0, 1, 0]), 0.0),
                pv.get(pcvl.BasicState([0, 0, 1]), 0.0),
            ]
        )
    np.testing.assert_allclose(np_preds, np.asarray(pv_preds), rtol=0.08, atol=0.08)


def test_validation_rejects_encoding_memristive_overlap() -> None:
    cfg = _two_moons_cfg(
        n_layers=1,
        encoding_phase_idx=(0, 1),
        n_enc_features=2,
        memristive_phase_idx=0,
    )
    with pytest.raises(ValueError, match="overlap"):
        validate_sim_config(cfg)


def test_validation_rejects_multi_layer_with_memristive() -> None:
    cfg = _two_moons_cfg(memristive_phase_idx=2, n_layers=2)
    with pytest.raises(ValueError, match="n_layers"):
        validate_sim_config(cfg)


def test_psr_gradient_trainable_phase_in_layer_two() -> None:
    cfg = _two_moons_cfg()
    validate_sim_config(cfg)
    n_ph = cfg.total_mesh_phases
    enc_set = set(cfg.encoding_slots)
    phase_idx = tuple(i for i in range(n_ph) if i not in enc_set)
    # Trainable phase in second layer (index >= 6)
    assert any(p >= 6 for p in phase_idx)

    rng = np.random.default_rng(3)
    theta0 = rng.uniform(0, 2 * np.pi, size=n_ph)
    x = np.array([[0.2, 0.8], [0.5, 0.4], [0.9, 0.1]])
    enc = encode_2d_to_phases_multi(x, n_layers=2)
    y = np.array([0, 1, 0])

    def loss_np(th: np.ndarray) -> float:
        out = run_simulation_sequence_np(th, enc, cfg, return_class_probs=True)
        logits = out - out.max(axis=1, keepdims=True)
        probs = np.exp(logits)
        probs /= probs.sum(axis=1, keepdims=True)
        return float(-np.mean(np.log(probs[np.arange(3), y] + 1e-15)))

    eps = 1e-4
    p_idx = phase_idx[-1]
    num_grad = (loss_np(theta0 + eps * (np.arange(n_ph) == p_idx)) - loss_np(theta0)) / eps

    th_t = torch.tensor(theta0, dtype=torch.float64, requires_grad=True)
    feats = MemristorLossPSR.apply(
        th_t,
        torch.from_numpy(enc).double(),
        phase_idx,
        cfg,
    )
    logits = feats - feats.max(dim=1, keepdims=True).values
    probs = torch.softmax(logits, dim=1)
    loss = -torch.log(probs[torch.arange(3), torch.from_numpy(y).long()] + 1e-15).mean()
    loss.backward()
    assert th_t.grad is not None
    assert abs(float(th_t.grad[p_idx]) - num_grad) < 0.05
