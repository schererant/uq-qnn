import os
import sys
import warnings
from pathlib import Path

# Set project root and config path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / 'config' / 'config.yaml'

# Add project root to Python path
sys.path.append(str(PROJECT_ROOT))

from config import Config as configfile
from dataloader import get_data, quartic_data
from logger import ExperimentLogger
from models.uq_3x3 import MemristorModel

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")

# Load configuration
config = configfile.from_yaml(CONFIG_PATH)

def main():
    # Initialize logger, comment 
    logger = ExperimentLogger()

    # Load data
    X_train, y_train, X_test, y_test, _ = get_data(
        n_data=config.data.n_data, 
        sigma_noise_1=config.data.sigma_noise_1,
        datafunction=quartic_data  # TODO: change to config.data.datafunction
    )
    
    # Log parameters
    logger.log_parameters(config)
    
    # Initialize model
    model = MemristorModel(
        training_steps=config.training.steps,
        memory_depth=config.training.memory_depth, 
        learning_rate=config.training.learning_rate, 
        cutoff_dim=config.training.cutoff_dim, 
        stochastic=config.prediction.stochastic,
        samples=config.prediction.samples,
        variance=config.prediction.variance,                             
        logger=logger, 
        param_id=None
        )
    
    # Train model
    model.train(X_train, y_train, plot=True)
    
    # Predict
    model.predict(X_test, y_test, plot=True)
    
    # Evaluate
    model.evaluate()

if __name__ == "__main__":
    main()
