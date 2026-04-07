"""UQ-QNN: Uncertainty Quantification for Quantum Neural Networks

A modular package for training photonic quantum neural networks with uncertainty quantification.
"""

from . import (
    autograd,
    circuits,
    coincidence,
    data,
    hardware,
    logging_config,
    loss,
    numpy_backend,
    simulation,
    training,
    utils,
)

__version__ = "0.1.0"
__author__ = "UQ-QNN Team"

# Main imports for convenience
from .autograd import MemristorLossPSR, photonic_psr_coeffs_torch
from .circuit import PhotonicCircuit
from .circuit_visualization import display_circuit_annotated, save_circuit_annotated
from .circuits import (
    build_circuit,
    clements_circuit,
    encoding_circuit,
    get_mzi_modes_for_phase,
    memristor_circuit,
)
from .coincidence import (
    accidental_correction,
    apply_noise_to_outcomes,
    get_cc_labels,
    get_cc_mode_pairs,
    postselect_measurement,
    probs_to_coincidences,
    probs_to_singles,
    working_detectors_to_cc_indices,
)
from .config import CircuitConfig, SimConfig, validate_sim_config
from .data import (
    get_data,
    load_measurement_pickle,
    load_timetags_measurement,
    multi_modal_data,
    neg_quadratic_data,
    neg_qubic_data,
    quartic_data,
    sinusoid_data,
    step_function_data,
    timetags_to_probabilities,
)
from .experiment import Experiment
from .hardware import (
    IDEAL,
    LAB_6MODE,
    NOISY_PROTOTYPE,
    PROFILES,
    CompositeNoise,
    DarkCountNoise,
    GaussianNoise,
    HardwareBackend,
    HardwareProfile,
    NoiseModel,
    RealHardwareBackend,
    ShotNoise,
    TimingParams,
    get_backend,
    get_profile,
    register_backend,
)
from .logging_config import add_file_handler, get_logger, log_params, set_verbosity
from .loss import PhotonicModel
from .simulation import (
    SimulationLogger,
    run_simulation_sequence_np,
    uncertainty_forward_pass,
)
from .training import gradient_check, train_pytorch, train_pytorch_generic
from .utils import print_run_params, resolve_n_swipe

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
    # Circuits
    "encoding_circuit",
    "memristor_circuit",
    "clements_circuit",
    "build_circuit",
    "get_mzi_modes_for_phase",
    # Simulation
    "run_simulation_sequence_np",
    "SimulationLogger",
    "uncertainty_forward_pass",
    # Autograd
    "photonic_psr_coeffs_torch",
    "MemristorLossPSR",
    # Loss / Training
    "PhotonicModel",
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
