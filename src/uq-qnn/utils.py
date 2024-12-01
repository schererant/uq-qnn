from datetime import datetime
import numpy as np

def log_training_loss(filepath, step, loss, phase1, phase3, memristor_weight):
    """Log training step results to file"""
    with open(filepath, 'a') as f:
        f.write(f"Step {step:4d}: Loss = {loss:.4f}, "
                f"Phase1 = {float(phase1):.4f}, "
                f"Phase3 = {float(phase3):.4f}, "
                f"Weight = {float(memristor_weight):.4f}\n")
        
def log_prediction_results(x_test, y_test, memory_depth, phase1, phase3, memristor_weight, stochastic, samples, var, cutoff_dim):
    """Create log file for prediction results"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filepath = f"memristor_prediction_{timestamp}.txt"
    
    with open(log_filepath, 'w') as f:
        # Header
        f.write("=" * 80 + "\n")
        f.write("Memristor Prediction Log\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")
        
        # Parameters
        f.write("Model Parameters:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Memory Depth: {memory_depth}\n")
        f.write(f"Phase1: {float(phase1):.4f}\n")
        f.write(f"Phase3: {float(phase3):.4f}\n")
        f.write(f"Memristor Weight: {float(memristor_weight):.4f}\n")
        f.write(f"Cutoff Dimension: {cutoff_dim}\n\n")
        
        # Prediction Settings
        f.write("Prediction Settings:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Stochastic: {stochastic}\n")
        f.write(f"Number of Samples: {samples}\n")
        f.write(f"Variance: {var}\n")
        f.write(f"Test Set Size: {len(x_test)}\n\n")
        
        f.write("Predictions:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Index':>6} {'Target':>10} {'Prediction':>12} {'Uncertainty':>12}\n")
        f.write("-" * 80 + "\n")
        
    return log_filepath

def log_prediction_step(filepath, index, target, prediction, uncertainty=None):
    """Log individual prediction results"""
    with open(filepath, 'a') as f:
        if uncertainty is not None:
            f.write(f"{index:6d} {float(target):10.4f} {float(prediction):12.4f} {float(uncertainty):12.4f}\n")
        else:
            f.write(f"{index:6d} {float(target):10.4f} {float(prediction):12.4f} {'N/A':>12}\n")



def memristor_update_function(x, y1, y2):
    """
    Computes the memristor update based on current input x and past values y1 and y2.

    Interpretation:
    - x: Current input value.
    - y1: Previous output (at time t-1).
    - y2: Output before previous (at time t-2).

    This function models how the memristor's state changes over time,
    incorporating both current input and past outputs.
    """
    return 0.4 * y1 + 0.4 * y1 * y2 + 0.6 * x ** 3 + 0.1

def multiply_three_inputs(x1, x2, x3):
    """
    Multiplies three input values.

    Interpretation:
    - x1, x2, x3: Input values at times t, t-1, and t-2, respectively.

    This function models a target where the output is the product of three inputs.
    """
    return x1 * x2 * x3

def target_function(xt, xt1, xt2):
    """
    Computes the target output as a sinusoidal function based on current and past inputs.

    Interpretation:
    - xt: Current input value at time t.
    - xt1: Previous input value at time t-1.
    - xt2: Input value at time t-2.

    This function defines a smooth, sinusoidal target for the model to learn.
    """
    return np.sin(2 * np.pi * (xt + xt1 + xt2)) + 1

# def target_function(xt, xt1, xt2):
#     """
#     Computes the target output based on current and past inputs.

#     Interpretation:
#     - xt: Current input value (at time t).
#     - xt1: Previous input value (at time t-1).
#     - xt2: Input value before previous (at time t-2).

#     This function defines the desired output for the model to learn.
#     """
#     return 0.4 * xt1 + 0.4 * xt1 * xt2 + 0.6 * xt ** 3 + 0.1