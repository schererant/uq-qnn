"""UQ-QNN: Uncertainty Quantification for Quantum Neural Networks

A modular package for training photonic quantum neural networks with uncertainty quantification.
"""

from . import autograd
from . import circuits
from . import data
from . import loss
from . import simulation
from . import training
from . import utils

__version__ = "0.1.0"
__author__ = "UQ-QNN Team"

# Main imports for convenience
from .data import get_data, load_measurement_pickle, quartic_data
from .circuits import encoding_circuit, memristor_circuit, build_circuit
from .simulation import run_simulation_sequence_np, SimulationLogger
from .autograd import photonic_psr_coeffs_torch, MemristorLossPSR
from .loss import PhotonicModel
from .training import train_pytorch, train_pytorch_generic, gradient_check
from .utils import main, config

__all__ = [
    # Modules
    "autograd",
    "circuits", 
    "data",
    "loss",
    "simulation",
    "training",
    "utils",
    # Main functions
    "get_data",
    "load_measurement_pickle",
    "quartic_data",
    "encoding_circuit",
    "memristor_circuit", 
    "build_circuit",
    "run_simulation_sequence_np",
    "SimulationLogger",
    "photonic_psr_coeffs_torch",
    "MemristorLossPSR",
    "PhotonicModel",
    "train_pytorch",
    "train_pytorch_generic",
    "gradient_check",
    "main",
    "config",
]