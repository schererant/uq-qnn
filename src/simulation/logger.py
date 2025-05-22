from collections import Counter
import time
from typing import Dict, Any

class SimulationLogger:
    """
    Logger for tracking simulation statistics and timing.
    """
    
    def __init__(self):
        """Initialize the logger with empty counters."""
        self.call_count = 0
        self.total_time = 0.0
        self.samples_counter = Counter()
        self.circuit_call_count = 0
        self.circuit_total_time = 0.0
        
    def log(self, elapsed: float, n_samples: int) -> None:
        """
        Log a simulation sequence run.
        
        Args:
            elapsed (float): Time taken for the sequence
            n_samples (int): Number of samples used
        """
        self.call_count += 1
        self.total_time += elapsed
        self.samples_counter[n_samples] += 1
        
    def log_circuit(self, elapsed: float) -> None:
        """
        Log a single circuit simulation.
        
        Args:
            elapsed (float): Time taken for the circuit simulation
        """
        self.circuit_call_count += 1
        self.circuit_total_time += elapsed
        
    def get_stats(self) -> Dict[str, Any]:
        """
        Get current statistics.
        
        Returns:
            Dict[str, Any]: Dictionary of statistics
        """
        stats = {
            'sequence_runs': self.call_count,
            'total_sequence_time': self.total_time,
            'circuit_simulations': self.circuit_call_count,
            'total_circuit_time': self.circuit_total_time,
            'samples_used': dict(self.samples_counter)
        }
        
        if self.call_count > 0:
            stats['avg_sequence_time'] = self.total_time / self.call_count
            
        if self.circuit_call_count > 0:
            stats['avg_circuit_time'] = self.circuit_total_time / self.circuit_call_count
            
        return stats
        
    def report(self) -> None:
        """Print a formatted report of the simulation statistics."""
        stats = self.get_stats()
        
        print(f"[SimulationLogger] Circuit sequence runs: {stats['sequence_runs']}")
        print(f"[SimulationLogger] Total sequence time: {stats['total_sequence_time']:.3f} seconds")
        
        if 'avg_sequence_time' in stats:
            print(f"[SimulationLogger] Avg time per sequence: {stats['avg_sequence_time']:.6f} seconds")
            
        print(f"[SimulationLogger] Sampler sample counts used:")
        for n_samples, freq in stats['samples_used'].items():
            print(f"  {n_samples} samples: {freq} times")
            
        print(f"[SimulationLogger] Individual circuit simulations: {stats['circuit_simulations']}")
        print(f"[SimulationLogger] Total circuit sim time: {stats['total_circuit_time']:.3f} seconds")
        
        if 'avg_circuit_time' in stats:
            print(f"[SimulationLogger] Avg time per circuit sim: {stats['avg_circuit_time']:.6f} seconds") 