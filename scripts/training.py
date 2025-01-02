import unittest
import numpy as np
import tensorflow as tf
from strawberryfields.ops import *
import warnings
import csv
import os
import shutil
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataloader import get_data, quartic_data
from src.logger import ExperimentLogger
from src.config import Config, HyperparameterConfig, ModelComparisonConfig, MLPConfig, PolynomialConfig, PredictionConfig, TrainingConfig, DataConfig
from src.model import train_memristor
from src.modelmegabigding import train_megabigmemristor

tf.get_logger().setLevel('ERROR')
warnings.filterwarnings("ignore")

logger = ExperimentLogger()

config = Config.from_yaml('config/config.yaml')


# Get data
X_train, y_train, X_test, y_test, _ = get_data(n_data=config.data.n_data, 
                                                sigma_noise_1=config.data.sigma_noise_1, 
                                                datafunction=quartic_data
                                                )

# when running training creation of experiment report does not work yet. Haven't managed to find the bug
# Train model
res_mem, phase1, phase2, phase4, phase8, phase9, phase10, phase11, phase12 = train_megabigmemristor(
    X_train=X_train,
    y_train=y_train,
    memory_depth=config.training.memory_depth,
    training_steps=config.training.steps,
    learning_rate=config.training.learning_rate,
    cutoff_dim=config.training.cutoff_dim,
    logger=logger
)