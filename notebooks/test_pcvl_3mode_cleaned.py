import perceval as pcvl
from perceval import Circuit, BasicState, NoiseModel, Processor, PostSelect, pdisplay, Detector
from perceval.algorithm import Sampler
from perceval import catalog
from tqdm import tqdm  # For Jupyter notebook progress bars
import numpy as np
from typing import Callable

#set the random seed for reproducibility
np.random.seed(42)


def quartic_data(x):
    return np.power(x, 4)

def get_data(n_data: int = 100, sigma_noise_1: float = 0.0, datafunction: Callable = quartic_data):
    """Define a function based toy regression dataset using NumPy.

    Args:
      n_data: number of data points
      sigma_noise_1: injected sigma noise on targets
      datafunction: function to compute labels based on input data

    Returns:
      train_input, train_target, test_input, test_target, label_noise
    """
    x_min = 0
    x_max = 1
    X_train = np.linspace(x_min, x_max, n_data)
    
    # split training set
    gap_start = x_min + 0.35 * (x_max - x_min)
    gap_end = x_min + 0.6 * (x_max - x_min)

    # create label noise
    # Note: For reproducibility, consider setting np.random.seed() outside this function
    # or using a np.random.Generator instance.
    noise_1 = np.random.normal(0, 1, size=n_data) * sigma_noise_1
    noise_1 = np.where(X_train > gap_end, 0.0, noise_1)  # Only add noise to the left

    # create simple function based labels data set and
    # add gaussian noise
    label_noise = noise_1
    y_train = datafunction(X_train) + label_noise

    train_idx = (X_train < gap_end) & (X_train > gap_start)

    # update X_train
    X_train = X_train[~train_idx]
    y_train = y_train[~train_idx]
    # Also filter label_noise if it's intended to correspond to the filtered X_train/y_train
    # label_noise = label_noise[~train_idx] # Uncomment if needed

    # test over the whole line
    X_test = np.linspace(x_min, x_max, 500)
    y_test = datafunction(X_test)

    return X_train, y_train, X_test, y_test, label_noise


def encoding_circuit(encoded_phase):
    """Builds the encoding circuit using Perceval components."""
    # Assumes SF Rgate(phi) corresponds to pcvl PS(phi). Verify if needed.
    circuit = pcvl.Circuit(2, name="Encoding Circuit")
    # Input encoding MZI (modes 0, 1)
    circuit.add((0, 1), pcvl.BS())
    circuit.add((1,), pcvl.PS(phi=encoded_phase))
    circuit.add((0, 1), pcvl.BS())
    return circuit

def memristor_circuit(phase1, memristor_phase, phase3):
    memristor_circuit = pcvl.Circuit(3, name="Memristor Circuit")
    # First MZI (modes 0, 1)
    memristor_circuit.add((0, 1), pcvl.BS())
    memristor_circuit.add((1,), pcvl.PS(phi=phase1))
    memristor_circuit.add((0, 1), pcvl.BS())
    # Memristor MZI (modes 1, 2)
    memristor_circuit.add((1, 2), pcvl.BS())
    memristor_circuit.add((2,), pcvl.PS(phi=memristor_phase))
    memristor_circuit.add((1, 2), pcvl.BS())
    # Third MZI (modes 0, 1)
    memristor_circuit.add((0, 1), pcvl.BS())
    memristor_circuit.add((1,), pcvl.PS(phi=phase3))
    memristor_circuit.add((0, 1), pcvl.BS())
    
    return memristor_circuit


def build_circuit(phase1, memristor_phase, phase3, encoded_phase):
    """Builds the 3-mode memristor circuit using Perceval components."""
    # Assumes SF Rgate(phi) corresponds to pcvl PS(phi). Verify if needed.
    circuit = pcvl.Circuit(3, name="Full Memristor Circuit")
    # Encoding circuit
    circuit.add(0, encoding_circuit(encoded_phase))
    circuit.add(0, memristor_circuit(phase1, memristor_phase, phase3))
    
    return circuit
    

def run_simulation_sequence(params, encoded_phases_all, memory_depth, plot_circuit=False):
    """Runs the simulation over the sequence, updating memory."""
    phase1, phase3, memristor_weight = params
    num_samples = len(encoded_phases_all)

    memory_p1 = np.zeros(memory_depth, dtype=np.float64)
    memory_p2 = np.zeros(memory_depth, dtype=np.float64)

    predictions_001 = np.zeros(num_samples, dtype=np.float64)
    # predictions_010 = np.zeros(num_samples, dtype=np.float64) # Only needed internally

    backend = "SLOS"  # Efficient backend for Fock states
    input_state = pcvl.BasicState([0, 1, 0])
    state_001 = pcvl.BasicState([0, 0, 1])
    state_010 = pcvl.BasicState([0, 1, 0])


    # for i in tqdm(range(num_samples), desc='Running circuit simulation'):
    for i in range(num_samples):
        time_step = i % memory_depth

        if i == 0:
            memristor_phase = np.pi / 4.0 # acos(sqrt(0.5))
        else:
            # Calculate memristor phase based on memory
            mem_term1 = np.sum(memory_p1) / memory_depth
            mem_term2 = memristor_weight * np.sum(memory_p2) / memory_depth
            # Clip argument for numerical stability before sqrt and acos
            sqrt_arg = np.clip(mem_term1 + mem_term2, 1e-9, 1.0 - 1e-9)
            memristor_phase = np.arccos(np.sqrt(sqrt_arg))

        circuit = build_circuit(phase1, memristor_phase, phase3, encoded_phases_all[i])
        
        if plot_circuit:
            pcvl.pdisplay(circuit, show=True, output_format=pcvl.Format.TEXT, recursive=True)
            
        # Create a processor
        proc = pcvl.Processor(backend, circuit)
        proc.with_input(input_state)
        
        sampler = Sampler(proc)
        counts = sampler.sample_count(1000)
        probs = sampler.probs(1000)

        prob_state_001 = probs['results'].get(state_001, 0.0)
        prob_state_010 = probs['results'].get(state_010, 0.0)

        predictions_001[i] = prob_state_001
        # predictions_010[i] = prob_state_010 # Store if needed outside

        # Update memory
        memory_p1[time_step] = prob_state_010
        memory_p2[time_step] = prob_state_001

    # Return only the prediction needed for loss calculation
    return predictions_001


def mse_loss(y_true, y_pred):
    """Computes Mean Squared Error loss."""
    return np.mean(np.square(np.asarray(y_true) - np.asarray(y_pred)))

def objective_function(params, encoded_phases_all, y_train, memory_depth, plot_circuit=False):
    """Calculates the loss for a given set of parameters."""
    predictions_001 = run_simulation_sequence(params, encoded_phases_all, memory_depth, plot_circuit=plot_circuit)
    return mse_loss(y_train, predictions_001)