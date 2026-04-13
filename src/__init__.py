"""UQ-QNN: Uncertainty Quantification for Quantum Neural Networks.

A modular package for training photonic quantum neural networks with
uncertainty quantification.
"""

from __future__ import annotations

from importlib import import_module
from types import ModuleType

__version__ = "0.1.0"
__author__ = "UQ-QNN Team"

_MODULE_EXPORTS = {
    "autograd": "autograd",
    "circuits": "circuits",
    "coincidence": "coincidence",
    "data": "data",
    "hardware": "hardware",
    "loss": "loss",
    "logging_config": "logging_config",
    "numpy_backend": "numpy_backend",
    "simulation": "simulation",
    "training": "training",
    "utils": "utils",
}

_ATTR_EXPORTS = {
    "MemristorLossPSR": ("autograd", "MemristorLossPSR"),
    "photonic_psr_coeffs_torch": ("autograd", "photonic_psr_coeffs_torch"),
    "PhotonicCircuit": ("circuit", "PhotonicCircuit"),
    "display_circuit_annotated": (
        "circuit_visualization",
        "display_circuit_annotated",
    ),
    "save_circuit_annotated": ("circuit_visualization", "save_circuit_annotated"),
    "build_circuit": ("circuits", "build_circuit"),
    "clements_circuit": ("circuits", "clements_circuit"),
    "encoding_circuit": ("circuits", "encoding_circuit"),
    "get_mzi_modes_for_phase": ("circuits", "get_mzi_modes_for_phase"),
    "memristor_circuit": ("circuits", "memristor_circuit"),
    "accidental_correction": ("coincidence", "accidental_correction"),
    "apply_noise_to_outcomes": ("coincidence", "apply_noise_to_outcomes"),
    "get_cc_labels": ("coincidence", "get_cc_labels"),
    "get_cc_mode_pairs": ("coincidence", "get_cc_mode_pairs"),
    "postselect_measurement": ("coincidence", "postselect_measurement"),
    "probs_to_coincidences": ("coincidence", "probs_to_coincidences"),
    "probs_to_singles": ("coincidence", "probs_to_singles"),
    "working_detectors_to_cc_indices": (
        "coincidence",
        "working_detectors_to_cc_indices",
    ),
    "CircuitConfig": ("config", "CircuitConfig"),
    "SimConfig": ("config", "SimConfig"),
    "validate_sim_config": ("config", "validate_sim_config"),
    "get_data": ("data", "get_data"),
    "gaussian_bump_data": ("data", "gaussian_bump_data"),
    "load_measurement_pickle": ("data", "load_measurement_pickle"),
    "load_timetags_measurement": ("data", "load_timetags_measurement"),
    "multi_modal_data": ("data", "multi_modal_data"),
    "neg_quadratic_data": ("data", "neg_quadratic_data"),
    "neg_qubic_data": ("data", "neg_qubic_data"),
    "quartic_data": ("data", "quartic_data"),
    "sinusoid_data": ("data", "sinusoid_data"),
    "step_function_data": ("data", "step_function_data"),
    "timetags_to_probabilities": ("data", "timetags_to_probabilities"),
    "Experiment": ("experiment", "Experiment"),
    "IDEAL": ("hardware", "IDEAL"),
    "LAB_6MODE": ("hardware", "LAB_6MODE"),
    "NOISY_PROTOTYPE": ("hardware", "NOISY_PROTOTYPE"),
    "PROFILES": ("hardware", "PROFILES"),
    "CompositeNoise": ("hardware", "CompositeNoise"),
    "DarkCountNoise": ("hardware", "DarkCountNoise"),
    "GaussianNoise": ("hardware", "GaussianNoise"),
    "HardwareBackend": ("hardware", "HardwareBackend"),
    "HardwareProfile": ("hardware", "HardwareProfile"),
    "NoiseModel": ("hardware", "NoiseModel"),
    "RealHardwareBackend": ("hardware", "RealHardwareBackend"),
    "ShotNoise": ("hardware", "ShotNoise"),
    "TimingParams": ("hardware", "TimingParams"),
    "get_backend": ("hardware", "get_backend"),
    "get_profile": ("hardware", "get_profile"),
    "register_backend": ("hardware", "register_backend"),
    "add_file_handler": ("logging_config", "add_file_handler"),
    "get_logger": ("logging_config", "get_logger"),
    "log_params": ("logging_config", "log_params"),
    "set_verbosity": ("logging_config", "set_verbosity"),
    "PhotonicModel": ("loss", "PhotonicModel"),
    "TrainedPhotonicState": ("loss", "TrainedPhotonicState"),
    "SimulationLogger": ("simulation", "SimulationLogger"),
    "run_simulation_sequence": ("simulation", "run_simulation_sequence"),
    "run_simulation_sequence_np": ("simulation", "run_simulation_sequence_np"),
    "uncertainty_forward_pass": ("simulation", "uncertainty_forward_pass"),
    "gradient_check": ("training", "gradient_check"),
    "train_pytorch": ("training", "train_pytorch"),
    "train_pytorch_generic": ("training", "train_pytorch_generic"),
    "print_run_params": ("utils", "print_run_params"),
    "resolve_n_swipe": ("utils", "resolve_n_swipe"),
}

__all__ = [
    # Modules
    "autograd",
    "circuits",
    "coincidence",
    "data",
    "hardware",
    "loss",
    "logging_config",
    "numpy_backend",
    "simulation",
    "training",
    "utils",
    # Data
    "get_data",
    "load_measurement_pickle",
    "load_timetags_measurement",
    "timetags_to_probabilities",
    "quartic_data",
    "neg_quadratic_data",
    "neg_qubic_data",
    "sinusoid_data",
    "gaussian_bump_data",
    # Circuits
    "encoding_circuit",
    "memristor_circuit",
    "clements_circuit",
    "build_circuit",
    "get_mzi_modes_for_phase",
    # Simulation
    "run_simulation_sequence",
    "run_simulation_sequence_np",
    "SimulationLogger",
    "uncertainty_forward_pass",
    # Autograd
    "photonic_psr_coeffs_torch",
    "MemristorLossPSR",
    # Loss / Training
    "PhotonicModel",
    "TrainedPhotonicState",
    "train_pytorch",
    "train_pytorch_generic",
    "gradient_check",
    "validate_sim_config",
    # Config / Experiment
    "Experiment",
    "SimConfig",
    "CircuitConfig",
    "PhotonicCircuit",
    # Hardware
    "HardwareProfile",
    "HardwareBackend",
    "RealHardwareBackend",
    "NoiseModel",
    "GaussianNoise",
    "ShotNoise",
    "DarkCountNoise",
    "CompositeNoise",
    "TimingParams",
    "get_profile",
    "register_backend",
    "get_backend",
    "IDEAL",
    "LAB_6MODE",
    "NOISY_PROTOTYPE",
    "PROFILES",
    # Logging
    "get_logger",
    "set_verbosity",
    "add_file_handler",
    "log_params",
    # Utils
    "print_run_params",
    "resolve_n_swipe",
    # Visualization
    "display_circuit_annotated",
    "save_circuit_annotated",
    # Coincidence
    "get_cc_labels",
    "get_cc_mode_pairs",
    "working_detectors_to_cc_indices",
    "probs_to_singles",
    "probs_to_coincidences",
    "postselect_measurement",
    "apply_noise_to_outcomes",
    "accidental_correction",
]


def _load_module(name: str) -> ModuleType:
    module = import_module(f".{name}", __name__)
    globals()[name] = module
    return module


def __getattr__(name: str):
    if name in _MODULE_EXPORTS:
        return _load_module(_MODULE_EXPORTS[name])
    if name in _ATTR_EXPORTS:
        module_name, attr_name = _ATTR_EXPORTS[name]
        module = _load_module(module_name)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
