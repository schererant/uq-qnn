import time
import numpy as np
import perceval as pcvl
from perceval.algorithm import Sampler
from typing import Optional, Tuple, Sequence

from simulation.logger import SimulationLogger
from circuits import CircuitBase, FullCircuit

class SimulationRunner:
    """
    Handles quantum circuit simulation runs, supporting both discrete and continuous modes.
    """
    
    def __init__(self, circuit: CircuitBase, memory_depth: int = 2):
        """
        Initialize simulation runner.
        
        Args:
            circuit (CircuitBase): The quantum circuit to simulate
            memory_depth (int): Depth of the memory buffer
        """
        self.circuit = circuit
        self.memory_depth = memory_depth
        self.logger = SimulationLogger()
        
    def _validate_inputs(self, encoded_phases: np.ndarray, n_swipe: int, swipe_span: float) -> None:
        """Validate simulation inputs."""
        if encoded_phases is None:
            raise ValueError("encoded_phases must be provided")
        if n_swipe < 0:
            raise ValueError("n_swipe must be >= 0")
        if n_swipe > 0 and swipe_span <= 0:
            raise ValueError("swipe_span must be > 0 for continuous mode")
            
    def _run_single_circuit(self, 
                          params: Sequence[float],
                          enc_phi: float,
                          input_state: pcvl.BasicState,
                          state_001: pcvl.BasicState,
                          state_010: pcvl.BasicState,
                          n_samples: int) -> Tuple[float, float]:
        """Run a single circuit simulation and return probabilities."""
        if not isinstance(self.circuit, FullCircuit):
            raise TypeError("Circuit must be a FullCircuit instance")
            
        phi1, phi3, _ = params  # w is not used in circuit simulation
        t0 = time.perf_counter()
        
        # Build and run circuit
        circuit = self.circuit.build(phi1=phi1, mem_phi=np.pi/4, phi3=phi3, enc_phi=enc_phi)
        proc = self.circuit.get_processor(circuit, input_state)
        probs = Sampler(proc).probs(n_samples)["results"]
        
        self.logger.log_circuit(time.perf_counter() - t0)
        return probs.get(state_001, 0.0), probs.get(state_010, 0.0)
        
    def run_sequence(self,
                    params: np.ndarray,
                    encoded_phases: np.ndarray,
                    n_samples: int,
                    n_swipe: int = 0,
                    swipe_span: float = 0.0) -> np.ndarray:
        """
        Run a sequence of circuit simulations.
        
        Args:
            params (np.ndarray): [phi1, phi3, w] parameters
            encoded_phases (np.ndarray): Phase values to simulate
            n_samples (int): Number of samples for the Sampler
            n_swipe (int): Number of phase points per data point (0 for discrete)
            swipe_span (float): Total phase span for swiping
            
        Returns:
            np.ndarray: Predicted probabilities for each input point
        """
        self._validate_inputs(encoded_phases, n_swipe, swipe_span)
        start_time = time.perf_counter()
        
        # Setup states
        input_state = pcvl.BasicState([0, 1, 0])
        state_001 = pcvl.BasicState([0, 0, 1])
        state_010 = pcvl.BasicState([0, 1, 0])
        
        # Initialize memory and output arrays
        mem_p1 = np.zeros(self.memory_depth)
        mem_p2 = np.zeros(self.memory_depth)
        num_pts = len(encoded_phases)
        preds = np.zeros(num_pts)
        
        # Precompute offsets for continuous mode
        if n_swipe > 0:
            offsets = np.linspace(-swipe_span/2, swipe_span/2, n_swipe)
        
        # Main simulation loop
        for i in range(num_pts):
            t = i % self.memory_depth
            
            # Compute memory-driven phase
            if i == 0:
                mem_phi = np.pi / 4
            else:
                m1 = mem_p1.mean()
                m2 = mem_p2.mean()
                arg = np.clip(m1 + params[2] * m2, 1e-9, 1 - 1e-9)
                mem_phi = np.arccos(np.sqrt(arg))
            
            if n_swipe == 0:
                # Discrete mode
                p001, p010 = self._run_single_circuit(
                    params, encoded_phases[i], input_state, state_001, state_010, n_samples
                )
                preds[i] = p001
                mem_p1[t], mem_p2[t] = p010, p001
            else:
                # Continuous mode
                p1_swipe = np.empty(n_swipe)
                p2_swipe = np.empty(n_swipe)
                
                for k, off in enumerate(offsets):
                    enc_phi = encoded_phases[i] + off
                    p001, p010 = self._run_single_circuit(
                        params, enc_phi, input_state, state_001, state_010, n_samples
                    )
                    p1_swipe[k] = p010
                    p2_swipe[k] = p001
                
                preds[i] = p2_swipe.mean()
                mem_p1[t], mem_p2[t] = p1_swipe.mean(), p2_swipe.mean()
        
        elapsed = time.perf_counter() - start_time
        self.logger.log(elapsed, n_samples)
        return preds 