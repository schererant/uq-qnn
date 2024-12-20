import unittest
import tensorflow as tf
import numpy as np
import os

from src.dataloader import get_data


class TestGetData(unittest.TestCase):
    """Test cases for the get_data function."""

    @staticmethod
    def linear_test_function(x):
        """Simple linear function for testing."""
        return 2 * x + 1

    def setUp(self):
        """Set up test fixtures."""
        self.default_n_data = 100
        self.default_sigma = 0.0
        self.gap_start = 0.35
        self.gap_end = 0.6
        self.test_size = 500

    def test_output_shapes(self):
        """Test if the output shapes are correct."""
        X_train, y_train, X_test, y_test, noise = get_data(
            n_data=self.default_n_data,
            sigma_noise_1=self.default_sigma,
            datafunction=self.linear_test_function
        )
        
        # Check dimensions
        self.assertEqual(len(X_train.shape), 1, "X_train should be 1-dimensional")
        self.assertEqual(len(y_train.shape), 1, "y_train should be 1-dimensional")
        
        # Check matching lengths
        self.assertEqual(X_train.shape[0], y_train.shape[0], 
                        "X_train and y_train should have same length")
        
        # Check test set size
        self.assertEqual(X_test.shape[0], self.test_size, 
                        f"X_test should have {self.test_size} points")
        self.assertEqual(y_test.shape[0], self.test_size, 
                        f"y_test should have {self.test_size} points")

    def test_data_ranges(self):
        """Test if the data ranges are correct."""
        X_train, y_train, X_test, y_test, noise = get_data(
            n_data=self.default_n_data,
            sigma_noise_1=self.default_sigma
        )
        
        # Check X ranges
        self.assertGreaterEqual(tf.reduce_min(X_train), 0, 
                               "X_train minimum should be >= 0")
        self.assertLessEqual(tf.reduce_max(X_train), 1, 
                            "X_train maximum should be <= 1")
        self.assertGreaterEqual(tf.reduce_min(X_test), 0, 
                               "X_test minimum should be >= 0")
        self.assertLessEqual(tf.reduce_max(X_test), 1, 
                            "X_test maximum should be <= 1")

    def test_gap_existence(self):
        """Test if the gap region is properly excluded."""
        X_train, _, _, _, _ = get_data(
            n_data=self.default_n_data,
            sigma_noise_1=self.default_sigma
        )
        
        # Check for points in gap
        gap_mask = tf.logical_and(
            X_train >= self.gap_start,
            X_train <= self.gap_end
        )
        gap_points = tf.reduce_sum(tf.cast(gap_mask, tf.int32))
        
        self.assertEqual(gap_points.numpy(), 0, 
                        "There should be no points in the gap region")

    def test_noise_generation(self):
        """Test noise generation and application."""
        sigma = 0.1
        _, _, _, _, noise = get_data(
            n_data=1000,  # Larger dataset for better statistics
            sigma_noise_1=sigma,
            datafunction=self.linear_test_function
        )
        
        # Test noise statistics
        noise_mean = tf.reduce_mean(noise)
        noise_std = tf.math.reduce_std(noise)
        
        self.assertLess(abs(noise_mean), 0.1, 
                       "Noise mean should be close to 0")
        self.assertLess(abs(noise_std - sigma), 0.1, 
                       "Noise standard deviation should be close to sigma")

    def test_custom_datafunction(self):
        """Test with custom data function."""
        def custom_func(x):
            return x ** 2
        
        X_train, y_train, _, _, _ = get_data(
            n_data=self.default_n_data,
            sigma_noise_1=self.default_sigma,
            datafunction=custom_func
        )
        
        # Test function application
        expected_y = custom_func(X_train)
        max_diff = tf.reduce_max(tf.abs(y_train - expected_y))
        
        self.assertLess(max_diff, 1e-6, 
                       "Custom function not correctly applied")

    def test_different_sizes(self):
        """Test with different dataset sizes."""
        test_sizes = [10, 100, 1000]
        
        for n in test_sizes:
            X_train, y_train, X_test, y_test, _ = get_data(n_data=n)
            
            # Calculate approximate expected size after gap removal
            gap_fraction = self.gap_end - self.gap_start
            expected_size = int(n * (1 - gap_fraction))
            actual_size = len(X_train)
            
            # Allow for ±2 points difference due to discretization
            self.assertLess(abs(actual_size - expected_size), 3,
                          f"Unexpected training set size for n_data={n}")
            self.assertEqual(len(X_test), self.test_size,
                           "Test set size should always be 500")

    # def tearDown(self):
    #     """Clean up after each test method."""
    #     # Clean up the entire reports directory for the test experiment
    #     if os.path.exists(self.logger.base_dir):
    #         shutil.rmtree(self.logger.base_dir)
        

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)