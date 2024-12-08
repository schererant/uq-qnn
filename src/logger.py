import os
import json
import csv
from datetime import datetime
import tensorflow as tf
import pickle




def log_experiment_id(log_file_name, param_id, name="Experiment"):
    with open(log_file_name, "a") as f:
        f.write("=" * 80 + "\n")
        f.write(f"{name}_{param_id}\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")

def save_predictions(sample_id, predictions, filename='predictions.pkl'):
    """
    Save predictions to a pickle file using a dictionary structure.
    If the file exists, load existing data and update with new predictions.
    
    Args:
        sample_id: The sample identifier (e.g., 'sample1')
        predictions: The prediction data to save
        filename: Name of the pickle file
    """
    try:
        # Try to load existing data
        with open(filename, 'rb') as f:
            predictions_dict = pickle.load(f)
    except FileNotFoundError:
        # If file doesn't exist, start with empty dictionary
        predictions_dict = {}
    
    # Add or update predictions for this sample
    predictions_dict[sample_id] = predictions
    
    # Save updated dictionary
    with open(filename, 'wb') as f:
        pickle.dump(predictions_dict, f)

def load_predictions(filename='predictions.pkl'):
    """
    Load all predictions from the pickle file
    
    Returns:
        Dictionary of all saved predictions
    """
    with open(filename, 'rb') as f:
        return pickle.load(f)
































# class ExperimentLogger:
#     def __init__(self, experiment_name=None):
#         """Initialize experiment logger with timestamp and create directory structure."""
#         timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
#         self.experiment_name = experiment_name or "qnn_experiment"
#         self.base_dir = f"ml_logs/{self.experiment_name}_{timestamp}"
        
#         # Create directory structure
#         os.makedirs(f"{self.base_dir}/artifacts", exist_ok=True)
        
#         # Initialize log files
#         self.log_file = f"{self.base_dir}/experiment.log"
#         self.params_file = f"{self.base_dir}/parameters.json"
#         self.metrics_file = f"{self.base_dir}/metrics.csv"
#         self.model_summary = f"{self.base_dir}/model_summary.txt"
        
#         # Initialize metrics CSV with headers
#         with open(self.metrics_file, 'w', newline='') as f:
#             writer = csv.writer(f)
#             writer.writerow(['timestamp', 'phase', 'step', 'loss', 'phase1', 'phase3', 'memristor_weight'])

#     def log_parameters(self, config):
#         """Log experiment parameters to JSON file."""
#         params = {
#             'memory_depth': config.MEMORY_DEPTH,
#             'cutoff_dim': config.CUTOFF_DIM,
#             'training_steps': config.TRAINING_STEPS,
#             'training_learning_rate': config.TRAINING_LEARNING_RATE,
#             'predict_stochastic': config.PREDICT_STOCHASTIC,
#             'predict_samples': config.PREDICT_SAMPLES,
#             'predict_variance': config.PREDICT_VARIANCE,
#             'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#         }
        
#         with open(self.params_file, 'w') as f:
#             json.dump(params, f, indent=4)

#     def log_training_step(self, step, loss, phase1, phase3, memristor_weight):
#         """Log training metrics to both experiment.log and metrics.csv."""
#         timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
#         # Log to experiment.log
#         with open(self.log_file, 'a') as f:
#             f.write(f"[{timestamp}] Training Step {step}: Loss={loss:.4f}, "
#                    f"Phase1={float(phase1):.4f}, Phase3={float(phase3):.4f}, "
#                    f"Weight={float(memristor_weight):.4f}\n")
        
#         # Log to metrics.csv
#         with open(self.metrics_file, 'a', newline='') as f:
#             writer = csv.writer(f)
#             writer.writerow([timestamp, 'training', step, loss, float(phase1), 
#                            float(phase3), float(memristor_weight)])

#     def log_prediction(self, sample_idx, predictions, uncertainty=None):
#         """Log prediction results."""
#         timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
#         with open(self.log_file, 'a') as f:
#             f.write(f"[{timestamp}] Prediction Sample {sample_idx + 1}\n")
#             f.write(f"Mean prediction: {np.mean(predictions):.4f}\n")
#             if uncertainty is not None:
#                 f.write(f"Mean uncertainty: {np.mean(uncertainty):.4f}\n")

#     def log_final_results(self, final_metrics):
#         """Log final experiment results."""
#         with open(self.model_summary, 'w') as f:
#             f.write("=== Final Model Summary ===\n\n")
#             for metric_name, value in final_metrics.items():
#                 f.write(f"{metric_name}: {value}\n")

#     def save_model_artifact(self, model_data, filename):
#         """Save model artifacts."""
#         artifact_path = f"{self.base_dir}/artifacts/{filename}"
#         with open(artifact_path, 'wb') as f:
#             pickle.dump(model_data, f)

#     def log_exception(self, exception):
#         """Log any exceptions that occur during the experiment."""
#         timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#         with open(self.log_file, 'a') as f:
#             f.write(f"[{timestamp}] ERROR: {str(exception)}\n")