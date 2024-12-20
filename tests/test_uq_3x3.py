import unittest
import numpy as np
import tensorflow as tf
from src.models.uq_3x3 import MemristorModel
from src.logger import ExperimentLogger

# src/models/test_uq_3x3.py


class TestMemristorModel(unittest.TestCase):
    def setUp(self):
        # Sample parameters
        self.training_steps = 1
        self.memory_depth = 2
        self.learning_rate = 0.01
        self.cutoff_dim = 2
        self.stochastic = False
        self.samples = 1
        self.variance = 0.1
        self.logger = None  # Set to an instance of ExperimentLogger if needed
        self.param_id = 'test'

        # Initialize the model
        self.model = MemristorModel(
            training_steps=self.training_steps,
            memory_depth=self.memory_depth,
            learning_rate=self.learning_rate,
            cutoff_dim=self.cutoff_dim,
            stochastic=self.stochastic,
            samples=self.samples,
            variance=self.variance,
            logger=self.logger,
            param_id=self.param_id
        )

        # Sample data
        self.X_train = tf.constant(np.linspace(-1, 1, 5), dtype=tf.float64)
        self.y_train = tf.constant(np.linspace(0, 1, 5), dtype=tf.float64)
        self.X_test = self.X_train
        self.y_test = self.y_train

    def test_train(self):
        # Test the train method
        result = self.model.train(self.X_train, self.y_train, plot=False)
        self.assertIsNotNone(result)
        self.assertTrue(isinstance(result, dict))

    def test_predict_without_uncertainty(self):
        # Test the predict method
        self.model.predict(self.X_test, self.y_test, plot=False)
        self.assertIsNotNone(self.model.final_predictions)
        self.assertTrue(len(self.model.final_predictions) > 0)
        
    def test_predict_with_uncertainty(self):
        # Test the predict method with predictive uncertainty
        self.model.stochastic = True
        self.model.samples = 5
        self.model.predict(self.X_test, self.y_test, plot=False)
        self.assertIsNotNone(self.model.final_predictions)
        self.assertTrue(len(self.model.final_predictions) > 0)

    def test_evaluate(self):
        # Ensure that predict is called before evaluate
        self.model.predict(self.X_test, self.y_test, plot=False)
        metrics, categories = self.model.evaluate()
        self.assertIsNotNone(metrics)
        self.assertIsNotNone(categories)
        self.assertTrue(isinstance(metrics, dict))
        self.assertTrue(isinstance(categories, list))

if __name__ == '__main__':
    unittest.main()