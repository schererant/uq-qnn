# src/uq_qnn/logger.py
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np
import tensorflow as tf

class QNNLogger:
    """Logger for QNN experiments with file and dictionary storage."""
    
    def __init__(self, base_filename: str):
        """Initialize logger with base filename."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.txt_path = f"{base_filename}_{timestamp}.txt"
        self.pkl_path = f"{base_filename}_{timestamp}.pkl"
        self.data = {
            'timestamp': timestamp,
            'experiments': {},
            'hyperparameters': {},
            'results': {}
        }
        
        # Create initial log file
        with open(self.txt_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write(f"QNN Experiment Log - {timestamp}\n")
            f.write("=" * 80 + "\n\n")
    
    def _serialize_value(self, value: Any) -> Any:
        """Convert non-serializable values to serializable format."""
        if isinstance(value, tf.Variable):
            return float(value.numpy())
        elif isinstance(value, np.ndarray):
            return value.tolist()
        elif isinstance(value, dict):
            return {k: self._serialize_value(v) for k, v in value.items()}
        elif isinstance(value, (list, tuple)):
            return [self._serialize_value(x) for x in value]
        return value
    
    def log_hyperparameters(self, exp_id: str, **params):
        """Log hyperparameters for an experiment."""
        self.data['hyperparameters'][exp_id] = params
        
        with open(self.txt_path, 'a') as f:
            f.write(f"\nHyperparameters - {exp_id}\n")
            f.write("-" * 40 + "\n")
            for name, value in params.items():
                f.write(f"  {name}: {value}\n")
            f.write("\n")
    
    def log_training(self, exp_id: str, step: int, **metrics):
        """Log training step metrics."""
        if exp_id not in self.data['experiments']:
            self.data['experiments'][exp_id] = {'training': []}
            
        metrics['step'] = step
        metrics['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.data['experiments'][exp_id]['training'].append(
            {k: self._serialize_value(v) for k, v in metrics.items()}
        )
        
        with open(self.txt_path, 'a') as f:
            metrics_str = ", ".join(f"{k}={v:.4f}" if isinstance(v, float) 
                                  else f"{k}={v}" for k, v in metrics.items())
            f.write(f"Step {step:4d}: {metrics_str}\n")
    
    def log_results(self, exp_id: str, **results):
        """Log final results."""
        self.data['results'][exp_id] = {
            k: self._serialize_value(v) for k, v in results.items()
        }
        
        with open(self.txt_path, 'a') as f:
            f.write(f"\nResults - {exp_id}\n")
            f.write("-" * 40 + "\n")
            for name, value in results.items():
                f.write(f"  {name}: {value}\n")
            f.write("\n")
    
    def save(self):
        """Save data dictionary to pickle file."""
        with open(self.pkl_path, 'wb') as f:
            pickle.dump(self.data, f)