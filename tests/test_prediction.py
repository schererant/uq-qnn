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
from src.model import train_memristor, predict_memristor

tf.get_logger().setLevel('ERROR')
warnings.filterwarnings("ignore")

class TestPrediction(unittest.TestCase):
    def setUp(self):
        """Set up test environment."""
        # self.experiment_name = "test_experiment"
        self.logger = ExperimentLogger()
        self.config = Config(
            hyperparameter=HyperparameterConfig(enabled=True, steps_range=[10, 20], learning_rate_range=[0.01, 0.1], memory_depth_range=[5, 10], cutoff_dim_range=[5, 10]),
            model_comparison=ModelComparisonConfig(enabled=True, n_samples=[10, 20]),
            mlp=MLPConfig(hidden_layers=[10, 20], epochs=10, learning_rate=0.01),
            polynomial=PolynomialConfig(degree=2),
            prediction=PredictionConfig(selective_threshold=0.5, stochastic=True, samples=3, variance=0.1),
            training=TrainingConfig(memory_depth=5, cutoff_dim=5, steps=2, learning_rate=0.01),
            data=DataConfig(n_data=100, sigma_noise_1=0.1, datafunction="quartic_data")
        )
        self.X_train, self.y_train, self.X_test, self.y_test, _ = get_data(n_data=self.config.data.n_data, 
                                                sigma_noise_1=self.config.data.sigma_noise_1, 
                                                datafunction=quartic_data
                                                )
        # Skip training params
        self.phase1 = 1.5
        self.phase3 = 0.5
        self.memristor_weight = 0.3

    def test_predict_memristor_deterministic(self):
        predictions, targets, predictive_uncertainty, all_predictions = predict_memristor(self.X_test, 
                                                                self.y_test, 
                                                                memory_depth=self.config.training.memory_depth, 
                                                                phase1=self.phase1, 
                                                                phase3=self.phase3, 
                                                                memristor_weight=self.memristor_weight,
                                                                stochastic=False, 
                                                                var=self.config.prediction.variance, 
                                                                samples=self.config.prediction.samples,
                                                                cutoff_dim=self.config.training.cutoff_dim,
                                                                logger=self.logger
                                                                ) 
        # self.assertEqual(predictions.shape, self.y_test.shape)
        # self.assertEqual(targets.shape, self.y_test.shape)
        # self.assertEqual(predictive_uncertainty.shape, self.y_test.shape)
        # self.assertEqual(all_predictions.shape, (self.config.prediction.samples, self.y_test.shape[0]))
        # self.assertTrue(np.allclose(np.mean(all_predictions, axis=0), predictions, atol=1e-2))
        # self.assertTrue(np.allclose(np.var(all_predictions, axis=0), predictive_uncertainty, atol=1e-2))

            
    def test_predict_memristor_stochastic(self):
            predictions, targets, predictive_uncertainty, all_predictions = predict_memristor(self.X_test, 
                                                                    self.y_test, 
                                                                    memory_depth=self.config.training.memory_depth, 
                                                                    phase1=self.phase1, 
                                                                    phase3=self.phase3, 
                                                                    memristor_weight=self.memristor_weight,
                                                                    stochastic=True, 
                                                                    var=self.config.prediction.variance, 
                                                                    samples=self.config.prediction.samples,
                                                                    cutoff_dim=self.config.training.cutoff_dim,
                                                                    logger=self.logger
                                                                    )
            # self.assertEqual(predictions.shape, self.y_test.shape)
            # self.assertEqual(targets.shape, self.y_test.shape)
            # self.assertEqual(predictive_uncertainty.shape, self.y_test.shape)
            # self.assertEqual(all_predictions.shape, (self.config.prediction.samples, self.y_test.shape[0]))
            # self.assertTrue(np.allclose(np.mean(all_predictions, axis=0), predictions, atol=1e-2))
            # self.assertTrue(np.allclose(np.var(all_predictions, axis=0), predictive_uncertainty, atol=1e-2))

    def tearDown(self):
        """Clean up after each test method."""
        # Clean up the entire reports directory for the test experiment
        if os.path.exists(self.logger.base_dir):
            shutil.rmtree(self.logger.base_dir)
        
if __name__ == '__main__':
    unittest.main()