import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
# Set the working directory to the project root
project_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.config import Config

# Create default config
# DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), '../config/config.yaml')
Config = Config.from_yaml(os.path.join(project_root, 'config/config.yaml'))

import numpy as np
import tensorflow as tf
import strawberryfields as sf
from strawberryfields.ops import *
import pickle
import random as rd
import warnings
from datetime import datetime
import uncertainty_toolbox as uct
from itertools import product
from tqdm import tqdm

from src.dataloader import get_data, quartic_data
from src.plotting import plot_predictions, plot_training_results, plot_predictions_new, plot_eval_metrics, plot_mlp_architecture_vs_rmse, plot_data, plot_selective_predictions
from src.baseline import train_mlp_baseline, predict_mlp_baseline, train_polynomial_baseline, predict_polynomial_baseline
from src.uq import selective_prediction, compute_eval_metrics
# from src.model import train_memristor, predict_memristor, build_circuit
from src.utils import format_metrics, format_hyperparameters
from src.config import Config
from src.logger import ExperimentLogger
from src.model import train_memristor, predict_memristor


tf.get_logger().setLevel('ERROR')
warnings.filterwarnings("ignore")

logger = ExperimentLogger()

config = Config.from_yaml('../config/config.yaml')
param_id = f"_s{config.training.steps}_lr{config.training.learning_rate}_md{config.training.memory_depth}_cd{config.training.cutoff_dim}"

# Get data
X_train, y_train, X_test, y_test, _ = get_data(n_data=config.data.n_data, 
                                                sigma_noise_1=config.data.sigma_noise_1, 
                                                datafunction=quartic_data
                                                )

logger.log_parameters(config)

# Train model
res_mem, phase1, phase3, memristor_weight = train_memristor(X_train, 
                                                            y_train, 
                                                            memory_depth=config.training.memory_depth, 
                                                            training_steps=config.training.steps,
                                                            learning_rate=config.training.learning_rate,
                                                            cutoff_dim=config.training.cutoff_dim,
                                                            logger=logger,
                                                            param_id=param_id
                                                            )


artifact_path = f"{logger.base_dir}/artifacts/deterministic_predictions.pkl"

final_predictions, targets, predictive_uncertainty, all_predictions = predict_memristor(
    X_test,
    y_test,
    memory_depth=config.training.memory_depth,
    cutoff_dim=config.training.cutoff_dim,
    memristor_weight=memristor_weight,
    phase1=phase1,
    phase3=phase3,
    logger=logger,
    param_id=param_id,
    stochastic=False,
    var=config.prediction.variance,
    samples=config.prediction.samples
)

metrics, metric_categories = compute_eval_metrics(final_predictions, targets, predictive_uncertainty, logger, param_id)

model_data = {
    'final_predictions': final_predictions,
    'targets': targets,
    'predictive_uncertainty': predictive_uncertainty,
    'all_predictions': all_predictions,
    'metrics': metrics,
    'metric_categories': metric_categories,
    'config': config,
    'logger': logger,
}

with open(artifact_path, 'wb') as f:
    pickle.dump(model_data, f)