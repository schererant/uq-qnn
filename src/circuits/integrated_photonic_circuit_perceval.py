import perceval as pcvl
import numpy as np
from perceval.algorithm import Analyzer

# Configuration
CONFIG = {
    "CIRCUIT_SIZE": 6,  # Number of modes in the circuit
    "DEFAULT_INPUT_FOCK": [1, 0, 1, 0, 1, 0],  # Default input Fock state
    "DEFAULT_PHASES": None,  # Default phases (None for random initialization)
    "VISUALIZE": False,  # Whether to display the circuit
    "SIMULATION_BACKEND": "SLOS"  # Backend for simulation
}

class IntegratedPhotonicCircuitPerceval:
    """Perceval implementation of integrated photonic circuits."""
    
    def __init__(self, size=CONFIG["CIRCUIT_SIZE"], phases=CONFIG["DEFAULT_PHASES"], 
                 input_fock=CONFIG["DEFAULT_INPUT_FOCK"]):
        """
        Parameters:
            size (int): Number of modes (e.g., 6 for a 6x6 circuit).
            phases (List[float], optional): List of phases for the circuit.
            input_fock (List[int]): Input Fock state (0 or 1 per mode).
        """
        if len(input_fock) != size:
            raise ValueError(f"Input Fock state must have length {size}")
        if not all(x in [0, 1] for x in input_fock):
            raise ValueError("Input Fock state must contain only 0s and 1s")
            
        self.size = size
        self.input_fock = input_fock
        self.circuit = pcvl.Circuit(size)
         
        # Initialize parameters if not provided
        if phases is None:
            self.parameters = []
            # For a size x size Clements layout, we need size*(size-1)/2 phases
            num_phases = size * (size - 1) // 2
            for i in range(num_phases):
                self.parameters.append(pcvl.Parameter(f"φ{i+1}", value=0))
        else:
            # Convert provided phases to Perceval parameters
            self.parameters = [pcvl.Parameter(f"φ{i+1}", value=phase) for i, phase in enumerate(phases)]
    
    def build_clements_circuit(self):
        """Build a Clements-style rectangular mesh interferometer."""
        param_idx = 0
        
        # Implement the rectangular mesh (Clements layout)
        for layer in range(self.size):
            start_idx = layer % 2  # Alternating between even and odd layers
            
            for i in range(start_idx, self.size - 1, 2):
                # Add a Mach-Zehnder interferometer
                self.circuit.add((i, i+1), pcvl.BS())
                self.circuit.add((i+1,), pcvl.PS(phi=self.parameters[param_idx]))
                self.circuit.add((i, i+1), pcvl.BS())
                param_idx += 1
                
                if param_idx >= len(self.parameters):
                    break
        
        return self.circuit, self.input_fock

    def simulate(self, backend_name=CONFIG["SIMULATION_BACKEND"]):
        """
        Simulate the circuit and return the output distribution using Perceval's Analyzer.
        
        Parameters:
            backend_name (str): Name of the simulation backend to use.
            
        Returns:
            dict: Dictionary mapping output Fock states to their probabilities.
        """
        # Build the circuit if not already built
        if not hasattr(self, 'circuit') or self.circuit is None:
            self.build_clements_circuit()
            
        # Create processor and analyzer
        processor = pcvl.Processor(backend_name, self.circuit)
        analyzer = Analyzer(processor, [pcvl.BasicState(self.input_fock)], '*')
        
        # Get the distribution
        distribution = {}
        for state in analyzer.possible_outputs():
            prob = analyzer.distribution[state]
            distribution[tuple(state)] = prob
            
        return distribution

    def get_amplitude(self, output_state, backend_name=CONFIG["SIMULATION_BACKEND"]):
        """
        Get the probability amplitude for a specific output state.
        
        Parameters:
            output_state (List[int]): The output Fock state to compute amplitude for.
            backend_name (str): Name of the simulation backend to use.
            
        Returns:
            complex: The probability amplitude for the given output state.
        """
        if not hasattr(self, 'circuit') or self.circuit is None:
            self.build_clements_circuit()
            
        backend = pcvl.BackendFactory().get_backend(backend_name)
        backend.set_circuit(self.circuit)
        backend.set_input_state(pcvl.BasicState(self.input_fock))
        
        return backend.prob_amplitude(pcvl.BasicState(output_state))

    def get_probability(self, output_state, backend_name=CONFIG["SIMULATION_BACKEND"]):
        """
        Get the probability for a specific output state.
        
        Parameters:
            output_state (List[int]): The output Fock state to compute probability for.
            backend_name (str): Name of the simulation backend to use.
            
        Returns:
            float: The probability for the given output state.
        """
        if not hasattr(self, 'circuit') or self.circuit is None:
            self.build_clements_circuit()
            
        backend = pcvl.BackendFactory().get_backend(backend_name)
        backend.set_circuit(self.circuit)
        backend.set_input_state(pcvl.BasicState(self.input_fock))
        
        return backend.probability(pcvl.BasicState(output_state))

    def get_most_probable_outputs(self, n=5):
        """
        Get the n most probable output states from the simulation.
        
        Parameters:
            n (int): Number of most probable outputs to return.
            
        Returns:
            list: List of tuples (state, probability) sorted by probability.
        """
        distribution = self.simulate()
        sorted_states = sorted(distribution.items(), key=lambda x: x[1], reverse=True)
        return sorted_states[:n]


if __name__ == "__main__":
    # Create a 6x6 Clements interferometer with default configuration
    ipc = IntegratedPhotonicCircuitPerceval()
    circ, input_fock = ipc.build_clements_circuit()
    print(circ)
    
    # Visualize the circuit if configured
    if CONFIG["VISUALIZE"]:
        pcvl.pdisplay(circ)
        
    # Run simulation and get results
    print("\nInput Fock state:", input_fock)
    
    # Get some example amplitudes and probabilities
    example_output = [0, 1, 0, 1, 0, 1]
    print(f"\nAmplitude for output {example_output}: {ipc.get_amplitude(example_output)}")
    print(f"Probability for output {example_output}: {ipc.get_probability(example_output)}")
    
    # print("\nMost probable output states:")
    # for state, prob in ipc.get_most_probable_outputs():
    #     print(f"State {state}: {prob:.4f}")

    #
