import unittest
import numpy as np
import tensorflow as tf
from strawberryfields.ops import *
import warnings
import csv
import os
import shutil

from src.dataloader import get_data, quartic_data
from src.logger import ExperimentLogger
from src.config import Config, HyperparameterConfig, ModelComparisonConfig, MLPConfig, PolynomialConfig, PredictionConfig, TrainingConfig, DataConfig
from src.model import train_memristor

tf.get_logger().setLevel('ERROR')
warnings.filterwarnings("ignore")

class TestTraining(unittest.TestCase):
    def setUp(self):
        """Set up test environment."""
        # self.experiment_name = "test_experiment"
        self.logger = ExperimentLogger()
        self.config = Config(
            hyperparameter=HyperparameterConfig(enabled=True, steps_range=[10, 20], learning_rate_range=[0.01, 0.1], memory_depth_range=[5, 10], cutoff_dim_range=[5, 10]),
            model_comparison=ModelComparisonConfig(enabled=True, n_samples=[10, 20]),
            mlp=MLPConfig(hidden_layers=[10, 20], epochs=10, learning_rate=0.01),
            polynomial=PolynomialConfig(degree=2),
            prediction=PredictionConfig(selective_threshold=0.5, stochastic=True, samples=10, variance=0.1),
            training=TrainingConfig(memory_depth=5, cutoff_dim=5, steps=2, learning_rate=0.01),
            data=DataConfig(n_data=100, sigma_noise_1=0.1, datafunction="quartic_data")
        )
        self.X_train, self.y_train, self.X_test, self.y_test, _ = get_data(n_data=self.config.data.n_data, 
                                                sigma_noise_1=self.config.data.sigma_noise_1, 
                                                datafunction=quartic_data
                                                )

    def test_train_memristor(self):
        """Test training of memristor model."""
        self.logger.log_parameters(self.config)
        res_mem, phase1, phase3, memristor_weight = train_memristor(
            X_train=self.X_train,
            y_train=self.y_train,
            memory_depth=self.config.training.memory_depth,
            training_steps=self.config.training.steps,
            learning_rate=self.config.training.learning_rate,
            cutoff_dim=self.config.training.cutoff_dim,
            logger=self.logger
        )

        # Verify results
        self.assertIsInstance(res_mem, dict)
        self.assertIsInstance(phase1, tf.Variable)
        self.assertIsInstance(phase3, tf.Variable)
        self.assertIsInstance(memristor_weight, tf.Variable)

        # Check log file
        with open(self.logger.log_file, 'r') as f:
            log_content = f.read()
            self.assertIn("=== Experiment Parameters ===", log_content)
            self.assertIn("Training Step", log_content)
            
        # Check metrics file
        with open(self.logger.metrics_file, 'r') as f:
            csv_reader = csv.reader(f)
            headers = next(csv_reader)
            self.assertEqual(headers, ['timestamp', 'phase', 'step', 'loss', 'phase1', 'phase3', 'memristor_weight'])
            rows = list(csv_reader)
            self.assertEqual(len(rows), self.config.training.steps)  # One row per training step

    def tearDown(self):
        """Clean up after each test method."""
        # Clean up the entire reports directory for the test experiment
        if os.path.exists(self.logger.base_dir):
            shutil.rmtree(self.logger.base_dir)
        
        
if __name__ == '__main__':
    unittest.main()