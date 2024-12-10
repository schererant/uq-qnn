import os
import unittest
import json
import csv
# import sys
from datetime import datetime
from src.logger import ExperimentLogger
from src.config import Config, HyperparameterConfig, ModelComparisonConfig, MLPConfig, PolynomialConfig, PredictionConfig, TrainingConfig, DataConfig

# Adjust the Python path to include the root directory of your project
# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# if project_root not in sys.path:
#     sys.path.append(project_root)

class TestExperimentLogger(unittest.TestCase):
    def setUp(self):
        """Set up test environment."""
        self.experiment_name = "test_experiment"
        self.logger = ExperimentLogger(experiment_name=self.experiment_name)
        self.config = Config(
            hyperparameter=HyperparameterConfig(enabled=True, steps_range=[10, 20], learning_rate_range=[0.01, 0.1], memory_depth_range=[5, 10], cutoff_dim_range=[5, 10]),
            model_comparison=ModelComparisonConfig(enabled=True, n_samples=[10, 20]),
            mlp=MLPConfig(hidden_layers=[10, 20], epochs=10, learning_rate=0.01),
            polynomial=PolynomialConfig(degree=2),
            prediction=PredictionConfig(selective_threshold=0.5, stochastic=True, samples=3, variance=0.1),
            training=TrainingConfig(memory_depth=5, cutoff_dim=5, steps=10, learning_rate=0.01),
            data=DataConfig(n_data=100, sigma_noise_1=0.1, datafunction="quartic_data")
        )

    def tearDown(self):
        """Clean up test environment."""
        if os.path.exists(self.logger.base_dir):
            for root, dirs, files in os.walk(self.logger.base_dir, topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
            os.rmdir(self.logger.base_dir)

    def test_log_parameters(self):
        """Test logging of experiment parameters."""
        self.logger.log_parameters(self.config)
        self.assertTrue(os.path.exists(self.logger.params_file))
        with open(self.logger.params_file, 'r') as f:
            params = json.load(f)
        self.assertEqual(params['hyperparameter']['enabled'], True)
        self.assertEqual(params['training']['steps'], 10)

    def test_log_training_step(self):
        """Test logging of training steps."""
        step = 1
        loss = 0.1234
        phase1 = 1.234
        phase3 = 2.345
        memristor_weight = 0.567
        self.logger.log_training_step(step, loss, phase1, phase3, memristor_weight)
        self.assertTrue(os.path.exists(self.logger.metrics_file))
        with open(self.logger.metrics_file, 'r') as f:
            reader = csv.reader(f)
            rows = list(reader)
        self.assertEqual(len(rows), 2)  # Header + 1 log entry
        self.assertEqual(rows[1][1], 'training')
        self.assertEqual(float(rows[1][3]), loss)

if __name__ == '__main__':
    unittest.main()