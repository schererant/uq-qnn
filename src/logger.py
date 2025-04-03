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
    def __init__(self, experiment_name=None, load_existing=None, phases=None, weights=None):
        """Initialize experiment logger with timestamp and dynamic directory structure."""
        self.phases = phases or []
        self.weights = weights or []

        if load_existing:
            self.base_dir = load_existing
            self.experiment_name = os.path.basename(self.base_dir)
        else:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.experiment_name = experiment_name or "experiment"
            self.base_dir = f"reports/{self.experiment_name}_{timestamp}"

            os.makedirs(f"{self.base_dir}/plots", exist_ok=False)
            os.makedirs(f"{self.base_dir}/artifacts", exist_ok=False)

            self.log_file = f"{self.base_dir}/experiment.log"
            self.params_file = f"{self.base_dir}/parameters.json"
            self.metrics_file = f"{self.base_dir}/metrics.csv"
            self.model_summary = f"{self.base_dir}/model_summary.txt"
            self.trained_model = f"{self.base_dir}/trained_model.pkl"

            with open(self.log_file, 'w') as f:
                f.write(f"=== {self.experiment_name} Experiment ===\n\n")

            with open(self.model_summary, 'w') as f:
                f.write("=== Model Summary ===\n\n")

            # Dynamic CSV headers
            headers = ['timestamp', 'phase', 'step', 'loss']
            headers += [f"phase_{i+1}" for i in range(len(self.phases))]
            headers += [f"memristor_weight_{i+1}" for i in range(len(self.weights))]

            with open(self.metrics_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(headers)

            log_experiment_id(self.log_file, timestamp, name=self.experiment_name)

    def log_parameters(self, config):
        params = {section: getattr(config, section).__dict__ for section in dir(config) if not section.startswith('_')}
        params['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        with open(self.params_file, 'w') as f:
            json.dump(params, f, indent=4)

        with open(self.log_file, 'a') as f:
            f.write("=== Experiment Parameters ===\n")
            for section, section_params in params.items():
                f.write(f"{section}: {section_params}\n")
            f.write("\n")

    def log_initial_training_phase(self, phases, weights):
        with open(self.log_file, 'a') as f:
            phase_str = ', '.join([f"Phase{i+1}={phase:.4f}" for i, phase in enumerate(phases)])
            weight_str = ', '.join([f"Weight{i+1}={weight:.4f}" for i, weight in enumerate(weights)])
            f.write(f"Initial Training Phase: {phase_str}, {weight_str}\n")

    def log_training_step(self, step, loss, phases, weights):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        with open(self.log_file, 'a') as f:
            phase_str = ', '.join([f"Phase{i+1}={phase:.4f}" for i, phase in enumerate(phases)])
            weight_str = ', '.join([f"Weight{i+1}={weight:.4f}" for i, weight in enumerate(weights)])
            f.write(f"[{timestamp}] Training Step {step}: Loss={loss:.4f}, {phase_str}, {weight_str}\n")

        with open(self.metrics_file, 'a', newline='') as f:
            writer = csv.writer(f)
            row = [timestamp, 'training', step, loss] + phases + weights
            writer.writerow(row)

    def log_final_training_phase(self, phases, weights, loss):
        with open(self.log_file, 'a') as f:
            phase_str = ', '.join([f"Phase{i+1}={phase}" for i, phase in enumerate(phases)])
            weight_str = ', '.join([f"Weight{i+1}={weight}" for i, weight in enumerate(weights)])
            f.write(f"Final Training Phase: {phase_str}, {weight_str}\n")
            f.write(f"Final Loss: {loss:.4f}\n")

    def log_prediction_step(self, step, loss, phases, weights):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        with open(self.metrics_file, 'a', newline='') as f:
            writer = csv.writer(f)
            row = [timestamp, 'prediction', step, loss] + phases + weights
            writer.writerow(row)


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
        
