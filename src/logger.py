import os
import json
import csv
from datetime import datetime
import tensorflow as tf
import pickle
import numpy as np



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



def create_experiment_dir():
    date = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = f"reports/baseline_{date}"
    os.makedirs(f"{log_path}/logs", exist_ok=False)
    os.makedirs(f"{log_path}/plots", exist_ok=False)

class ExperimentLogger:
    def __init__(self, experiment_name=None, load_existing=None):
        """Initialize experiment logger with timestamp and create directory structure."""
        if load_existing:
            # Load existing experiment directory
            self.base_dir = load_existing
            self.experiment_name = os.path.basename(self.base_dir)
        else:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.experiment_name = experiment_name or "experiment"
            self.base_dir = f"reports/{self.experiment_name}_{timestamp}"

            os.makedirs(f"{self.base_dir}/plots", exist_ok=False)
            os.makedirs(f"{self.base_dir}/artifacts", exist_ok=False)

            
            # Initialize log files
            self.log_file = f"{self.base_dir}/experiment.log"
            self.params_file = f"{self.base_dir}/parameters.json"
            self.metrics_file = f"{self.base_dir}/metrics.csv"
            self.model_summary = f"{self.base_dir}/model_summary.txt"
            self.trained_model = f"{self.base_dir}/trained_model.pkl"

            # Initialize log files
            with open(self.log_file, 'w') as f:
                f.write(f"=== {self.experiment_name} Experiment ===\n\n")

            # Initialize model summary file
            with open(self.model_summary, 'w') as f:
                f.write("=== Model Summary ===\n\n")
            
            # Initialize metrics CSV with headers
            with open(self.metrics_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'phase', 'step', 'loss', 'phase1', 'phase3', 'memristor_weight'])

            # Log experiment ID
            log_experiment_id(self.log_file, timestamp, name=self.experiment_name)

    def log_parameters(self, config):
        """Log experiment parameters to JSON file."""
        params = {
            'hyperparameter': config.hyperparameter.__dict__,
            'model_comparison': config.model_comparison.__dict__,
            'mlp': config.mlp.__dict__,
            'polynomial': config.polynomial.__dict__,
            'prediction': config.prediction.__dict__,
            'training': config.training.__dict__,
            'data': config.data.__dict__,
            # 'paths': config.paths.__dict__,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        with open(self.params_file, 'w') as f:
            json.dump(params, f, indent=4)

        # Write them to the log file
        with open(self.log_file, 'a') as f:
            f.write("=== Experiment Parameters ===\n")
            for section, section_params in params.items():
                if isinstance(section_params, dict):
                    f.write(f"{section}:\n")
                    for key, value in section_params.items():
                        f.write(f"  {key}: {value}\n")
                else:
                    f.write(f"{section}: {section_params}\n")
            f.write("\n")


    def log_initial_training_phase(self, phase1, phase3, memristor_weight):
        """Log the initial training phase parameters."""
        with open(self.log_file, 'a') as f:
            f.write(f"Initial Training Phase: Phase1={float(phase1)}, Phase3={float(phase3)}, Weight={float(memristor_weight)}\n")
        
    def log_training_step(self, step, loss, phase1, phase3, memristor_weight):
        """Log training metrics to both experiment.log and metrics.csv."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Log to experiment.log
        with open(self.log_file, 'a') as f:
            f.write(f"[{timestamp}] Training Step {step}: Loss={float(loss):.4f}, "
                   f"Phase1={float(phase1):.4f}, Phase3={float(phase3):.4f}, "
                   f"Weight={float(memristor_weight):.4f}\n")
        
        # Log to metrics.csv
        with open(self.metrics_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, 'training', step, float(loss), float(phase1), 
                           float(phase3), float(memristor_weight)])

    def log_final_training_phase(self, phase1, phase3, memristor_weight, loss):
        """Log the final training phase parameters."""
        with open(self.log_file, 'a') as f:
            f.write(f"Final Training Phase: Phase1={phase1}, Phase3={phase3}, Weight={memristor_weight}\n")
            f.write(f"Final Loss: {loss:.4f}\n")

    def log_prediction_step(self, step, loss, phase1, phase3, memristor_weight):
        """Log prediction metrics to both experiment.log and metrics.csv."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Log to experiment.log
        # with open(self.log_file, 'a') as f:
        #     f.write(f"[{timestamp}] Prediction Sample {step}: Loss={float(loss):.4f}, "
        #            f"Phase1={float(phase1):.4f}, Phase3={float(phase3):.4f}, "
        #            f"Weight={float(memristor_weight):.4f}\n")
        
        # Log to metrics.csv
        with open(self.metrics_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, 'prediction', step, float(loss), float(phase1), 
                           float(phase3), float(memristor_weight)])

    def log_prediction(self, predictions, uncertainty=None, samples=1):
        """Log prediction results."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        with open(self.log_file, 'a') as f:
            f.write(f"[{timestamp}] Predictions: ")
            f.write(f"Mean prediction: {np.mean(predictions):.4f}, ")
            if uncertainty is not None:
                f.write(f"Mean uncertainty: {np.mean(uncertainty):.4f}\n")
            f.write("\n")

    def log_final_results(self, final_metrics):
        """Log final experiment results."""
        with open(self.model_summary, 'w') as f:
            f.write("=== Final Model Summary ===\n\n")
            for metric_name, value in final_metrics.items():
                f.write(f"{metric_name}: {value}\n")

    def save_model_artifact(self, model_data, filename):
        """Save model artifacts."""
        artifact_path = f"{self.base_dir}/artifacts/{filename}"
        with open(artifact_path, 'wb') as f:
            pickle.dump(model_data, f)

    def load_model_artifact(self, filename):
        """Load model artifacts."""
        artifact_path = f"{self.base_dir}/artifacts/{filename}"
        with open(artifact_path, 'rb') as f:
            return pickle.load(f)

    def log_exception(self, exception):
        """Log any exceptions that occur during the experiment."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(self.log_file, 'a') as f:
            f.write(f"[{timestamp}] ERROR: {str(exception)}\n")

    def log_evaluation_metrics(self, metrics, param_id):
        """Log evaluation metrics to experiment.log."""
        with open(self.log_file, 'a') as f:
            f.write(f"=== Evaluation Metrics {param_id} ===\n")
            for category, value in metrics.items():
                f.write(f"{category}: {value}\n")
            f.write("\n")
        
