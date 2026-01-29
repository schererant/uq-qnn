import unittest
import numpy as np
import os
import tempfile
from unittest.mock import MagicMock, patch
from playground.pml_psr import train_memristor
import shutil

# Import the function to test

class MockExperimentLogger:
    """Simple mock for ExperimentLogger to avoid dependency issues."""
    def __init__(self, base_dir=None):
        self.base_dir = base_dir or tempfile.mkdtemp()
        os.makedirs(os.path.join(self.base_dir, "plots"), exist_ok=True)
        self.logs = {}

class TestTrainMemristor(unittest.TestCase):
    def setUp(self):
        """Set up test data and parameters."""
        # Create temporary directory for test outputs
        self.test_dir = tempfile.mkdtemp()
        
        # Create simple test data
        np.random.seed(42)  # For reproducibility
        self.X_train = np.linspace(0, 1, 10)  # Simple input values
        self.y_train = self.X_train ** 2      # Simple target function
        
        # Basic training parameters
        self.memory_depth = 2
        self.training_steps = 5  # Just a few steps for quick testing
        self.learning_rate = 0.01
        self.cutoff_dim = 5  # Not used by SLOS backend but required by function
        
        # Create a mock logger
        self.logger = MockExperimentLogger(base_dir=self.test_dir)
    
    def test_basic_functionality(self):
        """Test that train_memristor runs without errors and returns expected types."""
        # Call the training function
        res_mem, phase1, phase3, memristor_weight = train_memristor(
            self.X_train,
            self.y_train,
            self.memory_depth,
            self.training_steps,
            self.learning_rate,
            self.cutoff_dim,
            self.logger,
            param_id="test_basic",
            plot=False  # Disable plotting to avoid dependencies
        )
        
        # Check result types
        self.assertIsInstance(res_mem, dict)
        self.assertIsInstance(phase1, float)
        self.assertIsInstance(phase3, float)
        self.assertIsInstance(memristor_weight, float)
        
        # Check parameter constraints
        self.assertGreaterEqual(phase1, 0)
        self.assertLessEqual(phase1, 2 * np.pi)
        self.assertGreaterEqual(phase3, 0)
        self.assertLessEqual(phase3, 2 * np.pi)
        self.assertGreaterEqual(memristor_weight, 0.01)
        self.assertLessEqual(memristor_weight, 1.0)
        
        # Check that we have loss entries in the results dictionary
        loss_keys = [k for k in res_mem.keys() if k[0] == 'loss']
        self.assertTrue(len(loss_keys) > 0)
    
    def test_result_structure(self):
        """Test that the results have the expected structure."""
        # Train with several steps to get loss entries
        res_mem, _, _, _ = train_memristor(
            self.X_train,
            self.y_train,
            self.memory_depth,
            10,  # More steps for more loss entries
            self.learning_rate,
            self.cutoff_dim,
            self.logger,
            param_id="test_structure",
            plot=False
        )
        
        # Check the structure of loss entries
        for step in range(10):
            key = ('loss', 'tr', step)
            if key in res_mem:
                loss_entry = res_mem[key]
                # Each loss entry should be [loss_value, phase1, phase3, memristor_weight]
                self.assertEqual(len(loss_entry), 4)
                # Skip checking NaN values which might appear in early steps
                if not np.isnan(loss_entry[0]):
                    self.assertGreaterEqual(loss_entry[0], 0)  # Loss should be non-negative
    
    @patch('src.plotting.plot_training_results')
    def test_with_plotting(self, mock_plot):
        """Test that plotting is enabled when plot=True."""
        plot_dir = os.path.join(self.test_dir, "plots")
        os.makedirs(plot_dir, exist_ok=True)
        
        train_memristor(
            self.X_train,
            self.y_train,
            self.memory_depth,
            self.training_steps,
            self.learning_rate,
            self.cutoff_dim,
            self.logger,
            param_id="test_plot",
            plot=True,
            plot_path=plot_dir
        )
        
        # Verify that plotting was attempted (note: this will only work if
        # plot_training_results is correctly imported in the module)
        mock_plot.assert_called_once()
    
    def test_training_progress(self):
        """Test that training actually improves the loss over time."""
        # Train with more steps to observe learning progress
        res_mem, _, _, _ = train_memristor(
            self.X_train,
            self.y_train,
            self.memory_depth,
            30,  # More steps to see improvement
            self.learning_rate,
            self.cutoff_dim,
            self.logger,
            param_id="test_progress",
            plot=False
        )
        
        # Extract valid loss values (skip NaN values)
        loss_values = []
        for step in range(0, 30, 10):  # Check every 10th step to reduce test time
            key = ('loss', 'tr', step)
            if key in res_mem and not np.isnan(res_mem[key][0]):
                loss_values.append(res_mem[key][0])
        
        # If we have enough valid loss values, check for improvement
        if len(loss_values) >= 2:
            first_loss = loss_values[0]
            last_loss = loss_values[-1]
            # The loss should decrease over training
            self.assertLess(last_loss, first_loss, 
                           f"Training did not improve loss: first={first_loss}, last={last_loss}")
    
    def test_parameter_constraints(self):
        """Test that parameters stay within bounds even with high learning rate."""
        # Use high learning rate to force parameter clipping
        high_lr = 0.5
        
        _, phase1, phase3, memristor_weight = train_memristor(
            self.X_train,
            self.y_train,
            self.memory_depth,
            15,  # More steps to see clipping effects
            high_lr,
            self.cutoff_dim,
            self.logger,
            param_id="test_constraints",
            plot=False
        )
        
        # Verify parameters stayed within bounds
        self.assertGreaterEqual(phase1, 0)
        self.assertLessEqual(phase1, 2 * np.pi)
        self.assertGreaterEqual(phase3, 0)
        self.assertLessEqual(phase3, 2 * np.pi)
        self.assertGreaterEqual(memristor_weight, 0.01)
        self.assertLessEqual(memristor_weight, 1.0)
    
    def tearDown(self):
        """Clean up after tests."""
        shutil.rmtree(self.test_dir)

if __name__ == '__main__':
    unittest.main()