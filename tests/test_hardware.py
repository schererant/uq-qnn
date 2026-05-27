"""Tests for the hardware abstraction layer (src/hardware.py)."""

import numpy as np
import pytest

from src.data import compute_n_swipe
from src.hardware import (
    IDEAL,
    LAB_6MODE,
    NOISY_PROTOTYPE,
    PROFILES,
    CompositeNoise,
    DarkCountNoise,
    GaussianNoise,
    HardwareProfile,
    RealHardwareBackend,
    ShotNoise,
    TimingParams,
    get_backend,
    get_profile,
    register_backend,
)

# ── Noise model tests ─────────────────────────────────────────


class TestGaussianNoise:
    def test_output_shape(self):
        noise = GaussianNoise(std=0.1)
        rng = np.random.default_rng(42)
        outcomes = np.array([0.3, 0.5, 0.2])
        working = (0, 1, 2)
        result = noise(outcomes, working, ["C0", "C1", "C2"], rng=rng)
        assert result.shape == outcomes.shape

    def test_clipped_and_normalized(self):
        noise = GaussianNoise(std=0.5)
        rng = np.random.default_rng(0)
        outcomes = np.array([0.01, 0.98, 0.01])
        working = (0, 1, 2)
        result = noise(outcomes, working, ["C0", "C1", "C2"], rng=rng)
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)
        assert abs(result.sum() - 1.0) < 1e-10

    def test_seed_reproducibility(self):
        noise = GaussianNoise(std=0.1)
        outcomes = np.array([0.4, 0.4, 0.2])
        working = (0, 1, 2)
        labels = ["C0", "C1", "C2"]
        r1 = noise(outcomes, working, labels, rng=np.random.default_rng(123))
        r2 = noise(outcomes, working, labels, rng=np.random.default_rng(123))
        np.testing.assert_array_equal(r1, r2)

    def test_per_channel_std(self):
        noise = GaussianNoise(std=(0.01, 0.5, 0.01))
        rng = np.random.default_rng(42)
        outcomes = np.array([0.33, 0.34, 0.33])
        working = (0, 1, 2)
        result = noise(outcomes, working, ["C0", "C1", "C2"], rng=rng)
        assert result.shape == outcomes.shape

    def test_to_dict(self):
        noise = GaussianNoise(std=0.05)
        d = noise.to_dict()
        assert d == {"type": "gaussian", "std": 0.05}


class TestShotNoise:
    def test_output_shape(self):
        noise = ShotNoise(n_samples=1000)
        rng = np.random.default_rng(42)
        outcomes = np.array([0.5, 0.3, 0.2])
        result = noise(outcomes, (0, 1, 2), ["C0", "C1", "C2"], rng=rng)
        assert result.shape == outcomes.shape

    def test_normalized(self):
        noise = ShotNoise(n_samples=100)
        rng = np.random.default_rng(42)
        outcomes = np.array([0.5, 0.3, 0.2])
        result = noise(outcomes, (0, 1, 2), ["C0", "C1", "C2"], rng=rng)
        assert abs(result.sum() - 1.0) < 1e-10

    def test_to_dict(self):
        noise = ShotNoise(n_samples=500)
        assert noise.to_dict() == {"type": "shot", "n_samples": 500}


class TestDarkCountNoise:
    def test_adds_baseline(self):
        noise = DarkCountNoise(rate_per_detector=0.1)
        rng = np.random.default_rng(42)
        outcomes = np.array([0.5, 0.5, 0.0])
        result = noise(outcomes, (0, 1, 2), ["C0", "C1", "C2"], rng=rng)
        # All channels should be > 0 after dark counts
        assert np.all(result > 0)
        assert abs(result.sum() - 1.0) < 1e-10

    def test_to_dict(self):
        noise = DarkCountNoise(rate_per_detector=0.01)
        assert noise.to_dict() == {"type": "dark_count", "rate_per_detector": 0.01}


class TestCompositeNoise:
    def test_chaining(self):
        noise = CompositeNoise(
            models=(
                GaussianNoise(std=0.01),
                DarkCountNoise(rate_per_detector=0.001),
            )
        )
        rng = np.random.default_rng(42)
        outcomes = np.array([0.5, 0.3, 0.2])
        result = noise(outcomes, (0, 1, 2), ["C0", "C1", "C2"], rng=rng)
        assert result.shape == outcomes.shape
        assert abs(result.sum() - 1.0) < 1e-10

    def test_to_dict(self):
        noise = CompositeNoise(
            models=(GaussianNoise(std=0.05), ShotNoise(n_samples=100))
        )
        d = noise.to_dict()
        assert d["type"] == "composite"
        assert len(d["models"]) == 2
        assert d["models"][0]["type"] == "gaussian"
        assert d["models"][1]["type"] == "shot"


# ── TimingParams tests ────────────────────────────────────────


class TestTimingParams:
    def test_compute_n_swipe_agrees_with_data(self):
        tp = TimingParams(
            t_phase_ms=10.0,
            f_laser_khz=50.0,
            det_window_us=10.0,
            max_swipe=21,
        )
        expected = compute_n_swipe(10.0, 50.0, 10.0, 21)
        assert tp.compute_n_swipe() == expected

    def test_to_dict(self):
        tp = TimingParams(
            t_phase_ms=5.0,
            f_laser_khz=100.0,
            det_window_us=5.0,
            max_swipe=41,
        )
        d = tp.to_dict()
        assert d["t_phase_ms"] == 5.0
        assert d["max_swipe"] == 41

    def test_frozen(self):
        tp = TimingParams(
            t_phase_ms=10.0,
            f_laser_khz=50.0,
            det_window_us=10.0,
            max_swipe=21,
        )
        with pytest.raises(AttributeError):
            setattr(tp, "t_phase_ms", 20.0)


# ── HardwareProfile tests ────────────────────────────────────


class TestHardwareProfile:
    def test_to_dict_round_trip(self):
        d = LAB_6MODE.to_dict()
        assert d["name"] == "lab_6mode"
        assert d["backend"] == "numpy"
        assert d["noise"]["type"] == "gaussian"
        assert d["timing"]["t_phase_ms"] == 10.0

    def test_ideal_has_no_noise(self):
        assert IDEAL.noise is None
        assert IDEAL.timing is None

    def test_apply_to_config_sets_backend(self):
        cfg = {"n_modes": 3, "lr": 0.05}
        merged = LAB_6MODE.apply_to_config(cfg)
        assert merged["sim_backend"] == "numpy"
        # Original should not be mutated
        assert "sim_backend" not in cfg

    def test_apply_to_config_explicit_wins(self):
        cfg = {"sim_backend": "perceval", "n_modes": 3}
        merged = LAB_6MODE.apply_to_config(cfg)
        assert merged["sim_backend"] == "perceval"

    def test_frozen(self):
        with pytest.raises(AttributeError):
            setattr(IDEAL, "name", "changed")


# ── Profile registry tests ───────────────────────────────────


class TestGetProfile:
    def test_known_profiles(self):
        for name in ["ideal", "lab_6mode", "noisy_prototype"]:
            p = get_profile(name)
            assert p.name == name

    def test_unknown_raises(self):
        with pytest.raises(KeyError, match="Unknown hardware profile"):
            get_profile("nonexistent_profile")


# ── Real hardware backend tests ──────────────────────────────


class TestRealHardwareBackend:
    def test_not_available(self):
        backend = RealHardwareBackend()
        assert backend.is_available() is False

    def test_raises_not_implemented(self):
        backend = RealHardwareBackend()
        with pytest.raises(NotImplementedError, match="not yet implemented"):
            backend.run_circuit(
                np.array([0.1, 0.2]),
                np.array([1.0, 1.5]),
                None,
            )

    def test_name(self):
        backend = RealHardwareBackend()
        assert backend.name == "real_hardware"


# ── Backend registry tests ───────────────────────────────────


class TestBackendRegistry:
    def test_register_and_get(self):
        backend = RealHardwareBackend()
        register_backend("test_hw", backend)
        assert get_backend("test_hw") is backend

    def test_get_unknown_raises(self):
        with pytest.raises(KeyError, match="Unknown hardware backend"):
            get_backend("totally_unknown_backend")
