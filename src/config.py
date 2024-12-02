# src/uq-qnn/config.py
import os
import yaml
from datetime import datetime
from typing import Any, Dict
from dataclasses import dataclass

@dataclass
class HyperparameterConfig:
    enabled: bool
    steps_range: list[int]
    learning_rate_range: list[float] 
    memory_depth_range: list[int]
    cutoff_dim_range: list[int]

@dataclass
class ModelComparisonConfig:
    enabled: bool
    n_samples: list[int]

@dataclass
class MLPConfig:
    hidden_layers: list[int]
    epochs: int
    learning_rate: float

@dataclass
class PolynomialConfig:
    degree: int

@dataclass
class PredictionConfig:
    selective_threshold: float
    stochastic: bool
    samples: int
    variance: float

@dataclass
class TrainingConfig:
    memory_depth: int
    cutoff_dim: int
    steps: int
    learning_rate: float

@dataclass
class DataConfig:
    n_data: int
    sigma_noise_1: float
    datafunction: str

@dataclass
class Config:
    hyperparameter: HyperparameterConfig
    model_comparison: ModelComparisonConfig
    mlp: MLPConfig
    polynomial: PolynomialConfig
    prediction: PredictionConfig
    training: TrainingConfig
    data: DataConfig
    log_file_name: str

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'Config':
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        return cls(
            hyperparameter=HyperparameterConfig(**config_dict['hyperparameter_optimization']),
            model_comparison=ModelComparisonConfig(**config_dict['model_comparison']),
            mlp=MLPConfig(**config_dict['mlp']), 
            polynomial=PolynomialConfig(**config_dict['polynomial']),
            prediction=PredictionConfig(**config_dict['prediction']),
            training=TrainingConfig(**config_dict['training']),
            data=DataConfig(**config_dict['data']),
            log_file_name=f"logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        )

    def save(self, path: str) -> None:
        """Save current configuration to yaml file"""
        config_dict = {
            'hyperparameter_optimization': self.hyperparameter.__dict__,
            'model_comparison': self.model_comparison.__dict__,
            'mlp': self.mlp.__dict__,
            'polynomial': self.polynomial.__dict__,
            'prediction': self.prediction.__dict__,
            'training': self.training.__dict__,
            'data': self.data.__dict__
        }
        with open(path, 'w') as f:
            yaml.dump(config_dict, f)

# Create default config
DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), '../config/config.yaml')
config = Config.from_yaml(DEFAULT_CONFIG_PATH)