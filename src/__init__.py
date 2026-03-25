"""UQ-QNN: Uncertainty Quantification for Quantum Neural Networks

A modular package for training photonic quantum neural networks with uncertainty quantification.
"""

from . import autograd
from . import circuits
from . import coincidence
from . import data
from . import loss
from . import numpy_backend
from . import simulation
from . import training
from . import utils

__version__ = "0.1.0"
__author__ = "UQ-QNN Team"

# Main imports for convenience
from .data import (
    get_data,
    load_measurement_pickle,
    load_timetags_measurement,
    timetags_to_probabilities,
    quartic_data,
    neg_quadratic_data,
    neg_qubic_data,
    sinusoid_data,
    multi_modal_data,
    step_function_data,
)
from .circuits import (
    encoding_circuit,
    encoding_circuit_parametric,
    memristor_circuit,
    clements_circuit,
    build_circuit,
    build_parametric_circuit,
    get_mzi_modes_for_phase,
)
from .simulation import (
    run_simulation_sequence_np,
    SimulationLogger,
    uncertainty_forward_pass,
)
from .autograd import photonic_psr_coeffs_torch, MemristorLossPSR
from .loss import PhotonicModel
from .training import train_pytorch, train_pytorch_generic, gradient_check
from .utils import main, config, print_run_params
from .circuit_visualization import display_circuit_annotated, save_circuit_annotated
from .coincidence import (
    get_cc_labels,
    get_cc_mode_pairs,
    working_detectors_to_cc_indices,
    probs_to_singles,
    probs_to_coincidences,
    postselect_measurement,
    apply_noise_to_outcomes,
    accidental_correction,
)

__all__ = [
    # Modules
    "autograd",
    "circuits",
    "data",
    "loss",
    "numpy_backend",
    "simulation",
    "training",
    "utils",
    # Main functions
    "get_data",
    "load_measurement_pickle",
    "quartic_data",
    "neg_quadratic_data",
    "neg_qubic_data",
    "sinusoid_data",
    "encoding_circuit",
    "memristor_circuit",
    "clements_circuit",
    "build_circuit",
    "build_parametric_circuit",
    "encoding_circuit_parametric",
    "get_mzi_modes_for_phase",
    "run_simulation_sequence_np",
    "SimulationLogger",
    "uncertainty_forward_pass",
    "photonic_psr_coeffs_torch",
    "MemristorLossPSR",
    "PhotonicModel",
    "train_pytorch",
    "train_pytorch_generic",
    "gradient_check",
    "main",
    "config",
    "print_run_params",
    "display_circuit_annotated",
    "save_circuit_annotated",
    "coincidence",
    "load_timetags_measurement",
    "timetags_to_probabilities",
    "get_cc_labels",
    "get_cc_mode_pairs",
    "working_detectors_to_cc_indices",
    "probs_to_singles",
    "probs_to_coincidences",
    "postselect_measurement",
    "apply_noise_to_outcomes",
    "accidental_correction",
]
