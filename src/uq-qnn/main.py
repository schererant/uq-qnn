#!/usr/bin/env python3

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

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

from dataloader import get_data, quartic_data
from plotting import plot_predictions
from baseline import train_mlp_baseline, predict_mlp_baseline, train_polynomial_baseline, predict_polynomial_baseline
from uq import selective_prediction, compute_eval_metrics
from model import train_memristor, predict_memristor, build_circuit

tf.get_logger().setLevel('ERROR')
warnings.filterwarnings("ignore")

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
rd.seed(42)

print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))


#TODO: try different functions
#TODO: store hyperparameter, variance, outputs etc. to show difference
#TODO: save outputs etc.  
#TODO: 010 pol , ause neg loglike as loss 
#TODO: 

###### HYPERPARAMETERS ######

class Config:

    LOG_FILE_NAME = f"logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

    HYPERPARAMETER_OPTIMIZATION = True
    HYPER_STEPS_RANGE = [5]
    HYPER_LEARNING_RATE_RANGE = [0.01]
    HYPER_MEMORY_DEPTH_RANGE = [3]
    HYPER_CUTOFF_DIM_RANGE = [4, 5]

    #TODO: Check for which params we have the same loss
    MLP_HIDDEN_LAYERS = [64, 64]
    MLP_EPOCHS = 100
    MLP_LEARNING_RATE = 0.01

    POLYNOMIAL_DEGREE = 3

    SELECTIVE_PREDICTION_THRESHOLD = 0.8

    MEMORY_DEPTH = 5
    CUTOFF_DIM = 4

    TRAINING_STEPS = 5
    TRAINING_LEARNING_RATE = 0.05

    PREDICT_STOCHASTIC = False
    PREDICT_SAMPLES = 1
    PREDICT_VARIANCE = 0.1

    GET_DATA_N_DATA = 200
    GET_DATA_SIGMA_NOISE_1 = 0.1
    GET_DATA_DATAFUNCTION = quartic_data



def model_comparison(X_train, y_train, X_test, y_test):
    all_results = {}
    
    # 1. MLP Baseline

    print("Training MLP Baseline Models...")
    mlp_architectures = [
        [32],              
        [64, 64],         # Matches MLP_HIDDEN_LAYERS
        [128, 64, 64],    
    ]
    
    for hidden_layers in mlp_architectures:
        mlp_model = train_mlp_baseline(
            X_train, 
            y_train, 
            hidden_layers=hidden_layers,
            epochs=Config.MLP_EPOCHS,
            learning_rate=Config.MLP_LEARNING_RATE
        )
        mlp_predictions = predict_mlp_baseline(mlp_model, X_test)
        metrics, metric_categories = compute_eval_metrics(
            np.array(mlp_predictions), 
            np.array(y_test), 
            np.array([])
        )
        all_results[f"mlp_{len(hidden_layers)}_layers"] = {
            "metrics": metrics,
            "categories": metric_categories
        }

    
    # 2. QNN Base Model
    
    print("Training QNN Base Model...")
    res_mem, phase1, phase3, memristor_weight = train_memristor(
        X_train, 
        y_train,
        memory_depth=Config.MEMORY_DEPTH,
        training_steps=Config.TRAINING_STEPS,
        learning_rate=Config.TRAINING_LEARNING_RATE,
        cutoff_dim=Config.CUTOFF_DIM
    )
    

    # 3. QNN with UQ
    
    for n_samples in [5, 20, 50, 100]:

        print(f"Predict QNN UQ Model with {n_samples} samples...")
        predictions, targets, predictive_uncertainty = predict_memristor(
            X_test, 
            y_test,
            memory_depth=Config.MEMORY_DEPTH,
            phase1=phase1,
            phase3=phase3,
            memristor_weight=memristor_weight,
            stochastic=True,
            samples=n_samples,
            var=Config.PREDICT_VARIANCE
        )
        
        # Full dataset metrics
        full_metrics, full_metric_categories = compute_eval_metrics(
            predictions, 
            targets, 
            predictive_uncertainty
        )
        all_results[f"qnn_uq_samples_{n_samples}"] = {
            "metrics": full_metrics,
            "categories": full_metric_categories
        }

        # Selective prediction
        for threshold in [0.7, 0.8, 0.9]:

            print(f"Selective Prediction with threshold {threshold}...")
            sel_predictions, sel_targets, sel_uncertainty, remaining_fraction = selective_prediction(
                predictions, 
                targets, 
                predictive_uncertainty, 
                threshold=threshold
            )
            
            sel_metrics, sel_metric_categories = compute_eval_metrics(
                sel_predictions, 
                sel_targets, 
                sel_uncertainty
            )
            
            all_results[f"qnn_selective_t{threshold}_s{n_samples}"] = {
                "metrics": sel_metrics,
                "categories": sel_metric_categories,
                "remaining_fraction": remaining_fraction
            }

    return all_results

def save_results_to_txt(all_results, filepath="results.txt"):
    """Save model comparison results to a formatted text file"""
    
    def format_metrics(metrics_dict, indent=0):
        lines = []
        indent_str = " " * indent
        for key, value in metrics_dict.items():
            if isinstance(value, dict):
                lines.append(f"{indent_str}{key}:")
                lines.extend(format_metrics(value, indent + 2))
            elif isinstance(value, (np.ndarray, list)):
                lines.append(f"{indent_str}{key}: {np.mean(value):.4f}")
            else:
                lines.append(f"{indent_str}{key}: {value}")
        return lines

    with open(filepath, "w") as f:
        # Write header
        f.write("=" * 80 + "\n")
        f.write("Model Comparison Results\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")

        # Write parameters
        f.write("Parameters:\n")
        f.write(f"Memory Depth: {Config.MEMORY_DEPTH}\n")
        f.write(f"Cutoff Dimension: {Config.CUTOFF_DIM}\n")
        f.write(f"Training Steps: {Config.TRAINING_STEPS}\n")
        f.write(f"Learning Rate: {Config.TRAINING_LEARNING_RATE}\n\n")

        # Write results for each model
        for model_name, result in all_results.items():
            f.write("-" * 40 + "\n")
            f.write(f"Model: {model_name}\n")
            f.write("-" * 40 + "\n")
            
            if "metrics" in result:
                f.write("\nMetrics:\n")
                metric_lines = format_metrics(result["metrics"], indent=2)
                f.write("\n".join(metric_lines))
            
            if "remaining_fraction" in result:
                f.write(f"\nRemaining Fraction: {result['remaining_fraction']:.2%}\n")
            
            f.write("\n")

def hyperparameter_optimization(X_train, y_train, X_test, y_test):
    """
    Performs hyperparameter optimization and stores results in a consistent format
    with model_comparison function, including metrics for all results.
    """
    # Define the ranges for the hyperparameters
    steps_range = Config.HYPER_STEPS_RANGE
    learning_rate_range = Config.HYPER_LEARNING_RATE_RANGE
    memory_depth_range = Config.HYPER_MEMORY_DEPTH_RANGE
    cutoff_dim_range = Config.HYPER_CUTOFF_DIM_RANGE

    best_loss = float('inf')
    best_params = None
    best_metrics = None
    best_categories = None
    all_results = {}

    # Calculate total number of combinations
    total_combinations = len(steps_range) * len(learning_rate_range) * len(memory_depth_range) * len(cutoff_dim_range)
    
    # Create a tqdm iterator with total number of combinations
    for steps, learning_rate, memory_depth, cutoff_dim in tqdm(
        product(steps_range, learning_rate_range, memory_depth_range, cutoff_dim_range),
        total=total_combinations,
        desc="Hyperparameter Optimization",
        unit="combination"
    ):

    # for steps, learning_rate, memory_depth, cutoff_dim in [(steps_range[0], learning_rate_range[0], memory_depth_range[0], cutoff_dim_range[0])]:
        print(f"Training with steps={steps}, learning_rate={learning_rate}, memory_depth={memory_depth}, cutoff_dim={cutoff_dim}")

        # Train the memristor model
        res_mem, phase1, phase3, memristor_weight = train_memristor(
            X_train, 
            y_train, 
            memory_depth=memory_depth, 
            training_steps=steps, 
            learning_rate=learning_rate, 
            cutoff_dim=cutoff_dim,
            filename=Config.LOG_FILE_NAME
        )

        # Predict using the trained model
        predictions, targets, predictive_uncertainty = predict_memristor(
            X_test, 
            y_test, 
            memory_depth=memory_depth, 
            phase1=phase1, 
            phase3=phase3, 
            memristor_weight=memristor_weight, 
            stochastic=False, 
            samples=1, 
            var=0.0, 
            cutoff_dim=cutoff_dim
        )

        # Compute evaluation metrics
        metrics, metric_categories = compute_eval_metrics(predictions, targets, predictive_uncertainty)
        
        # Create a unique identifier for this parameter combination
        param_id = f"qnn_hp_s{steps}_lr{learning_rate}_md{memory_depth}_cd{cutoff_dim}"
        
        # Store results in the same format as model_comparison

        all_results[param_id] = {
            "metrics": metrics,
            "categories": metric_categories,
            "hyperparameters": {
                "steps": steps,
                "learning_rate": learning_rate,
                "memory_depth": memory_depth,
                "cutoff_dim": cutoff_dim
            }
        }

        # Update best parameters if current loss is lower
        if metrics['accuracy']['rmse'] < best_loss:
            best_loss = metrics['accuracy']['rmse']
            best_params = (steps, learning_rate, memory_depth, cutoff_dim)
            best_metrics = metrics
            best_categories = metric_categories

    # Add best parameters summary to results with metrics
    if best_params:
        all_results["best_parameters"] = {
            "metrics": best_metrics,
            "categories": best_categories,
            "hyperparameters": {
                "steps": best_params[0],
                "learning_rate": best_params[1],
                "memory_depth": best_params[2],
                "cutoff_dim": best_params[3],
                "best_rmse": best_loss
            }
        }
        print(f"Best parameters: steps={best_params[0]}, learning_rate={best_params[1]}, "
              f"memory_depth={best_params[2]}, cutoff_dim={best_params[3]} with RMSE={best_loss}")
    else:
        print("No valid hyperparameter combinations found.")

    return best_params, all_results

def save_hyperparameter_results(all_results, filepath="hyperparameter_results.txt"):
    """Save hyperparameter optimization results to a formatted text file"""
    
    def format_metrics(metrics_dict, indent=0):
        lines = []
        indent_str = " " * indent
        for key, value in metrics_dict.items():
            if isinstance(value, dict):
                lines.append(f"{indent_str}{key}:")
                lines.extend(format_metrics(value, indent + 2))
            elif isinstance(value, (np.ndarray, list)):
                lines.append(f"{indent_str}{key}: {np.mean(value):.4f}")
            else:
                lines.append(f"{indent_str}{key}: {value}")
        return lines

    def format_hyperparameters(hyperparameters_dict, indent=0):
        lines = []
        indent_str = " " * indent
        for key, value in hyperparameters_dict.items():
            if isinstance(value, float):
                lines.append(f"{indent_str}{key}: {value:.6f}")
            else:
                lines.append(f"{indent_str}{key}: {value}")
        return lines

    with open(filepath, "w") as f:
        # Write header
        f.write("=" * 80 + "\n")
        f.write("Hyperparameter Optimization Results\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")

        # Write hyperparameter ranges
        f.write("Hyperparameter Ranges:\n")
        f.write(f"Steps Range: {Config.HYPER_STEPS_RANGE}\n")
        f.write(f"Learning Rate Range: {Config.HYPER_LEARNING_RATE_RANGE}\n")
        f.write(f"Memory Depth Range: {Config.HYPER_MEMORY_DEPTH_RANGE}\n")
        f.write(f"Cutoff Dimension Range: {Config.HYPER_CUTOFF_DIM_RANGE}\n\n")

        # First write the best parameters if they exist
        if "best_parameters" in all_results:
            f.write("=" * 40 + "\n")
            f.write("Best Parameters\n")
            f.write("=" * 40 + "\n")
            
            result = all_results["best_parameters"]
            
            f.write("\nHyperparameters:\n")
            hyperparameter_lines = format_hyperparameters(result["hyperparameters"], indent=2)
            f.write("\n".join(hyperparameter_lines))
            
            if "metrics" in result:
                f.write("\n\nMetrics:\n")
                metric_lines = format_metrics(result["metrics"], indent=2)
                f.write("\n".join(metric_lines))
            
            f.write("\n\n")

        # Write results for each parameter combination
        for model_name, result in all_results.items():
            if model_name == "best_parameters":
                continue
                
            f.write("-" * 40 + "\n")
            f.write(f"Model: {model_name}\n")
            f.write("-" * 40 + "\n")
            
            f.write("\nHyperparameters:\n")
            if "hyperparameters" in result:
                hyperparameter_lines = format_hyperparameters(result["hyperparameters"], indent=2)
                f.write("\n".join(hyperparameter_lines))
            
            if "metrics" in result:
                f.write("\n\nMetrics:\n")
                metric_lines = format_metrics(result["metrics"], indent=2)
                f.write("\n".join(metric_lines))
            
            f.write("\n\n")

    print(f"Results saved to {filepath}")

def main():

    

    # Get the data
    X_train, y_train, X_test, y_test, _ = get_data(n_data=Config.GET_DATA_N_DATA, 
                                                   sigma_noise_1=Config.GET_DATA_SIGMA_NOISE_1, 
                                                   datafunction=Config.GET_DATA_DATAFUNCTION
                                                   )
    

    if Config.HYPERPARAMETER_OPTIMIZATION:
        # Perform hyperparameter optimization
        best_params, results = hyperparameter_optimization(X_train, y_train, X_test, y_test)
        steps, learning_rate, memory_depth, cutoff_dim = best_params

        save_hyperparameter_results(results, f"hyperparameter_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

        # # Save hyperparameter optimization results to a text file
        # with open("hyperparameter_optimization_results.txt", "w") as file:
        #     for steps, learning_rate, memory_depth, cutoff_dim, loss in results:
        #         file.write(f"steps={steps}, learning_rate={learning_rate}, memory_depth={memory_depth}, cutoff_dim={cutoff_dim}, loss={loss}\n")

        Config.TRAINING_STEPS = steps
        Config.TRAINING_LEARNING_RATE = learning_rate
        Config.MEMORY_DEPTH = memory_depth
        Config.CUTOFF_DIM = cutoff_dim


        all_results = model_comparison(X_train, y_train, X_test, y_test)
        save_results_to_txt(all_results, f"model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")


    
    # Train the memristor model
    res_mem, phase1, phase3, memristor_weight = train_memristor(X_train, 
                                                                y_train, 
                                                                memory_depth=Config.MEMORY_DEPTH, 
                                                                training_steps=Config.TRAINING_STEPS,
                                                                learning_rate=Config.TRAINING_LEARNING_RATE,
                                                                cutoff_dim=Config.CUTOFF_DIM
                                                                )

    # Save training results
    with open(f"{Config.LOG_FILE_NAME}.pkl", "wb") as file:
        pickle.dump(res_mem, file)

    # Predict using the trained model
    predictions, targets, predictive_uncertainty = predict_memristor(X_test, 
                                                                    y_test, 
                                                                    memory_depth=Config.MEMORY_DEPTH, 
                                                                    phase1=phase1, 
                                                                    phase3=phase3, 
                                                                    memristor_weight=memristor_weight,
                                                                    stochastic=Config.PREDICT_STOCHASTIC, 
                                                                    var=Config.PREDICT_VARIANCE, 
                                                                    samples=Config.PREDICT_SAMPLES,
                                                                    cutoff_dim=Config.CUTOFF_DIM
                                                                    )

    # Ensure predictions and X_test have the same length
    assert len(predictions) == len(X_test), "Predictions and X_test must have the same length"

    # Convert predictions, targets, and predictive_uncertainty to NumPy arrays
    predictions = np.array(predictions)
    targets = np.array(targets)
    predictive_uncertainty = np.array(predictive_uncertainty)

    # Compute evaluation metrics for full predictions
    full_metrics, full_metric_categories = compute_eval_metrics(predictions, 
                                                                targets, 
                                                                predictive_uncertainty
                                                                )
    
    print("Full Prediction Metrics:")
    for category in full_metric_categories:
        print(f"{category}: {full_metrics[category]}")

    # Apply selective prediction
    sel_predictions, sel_targets, sel_uncertainty, remaining_fraction = selective_prediction(predictions, 
                                                                                            targets, 
                                                                                            predictive_uncertainty, 
                                                                                            threshold=Config.SELECTIVE_PREDICTION_THRESHOLD
                                                                                            )
    
    print(f"Remaining Fraction after Selective Prediction: {remaining_fraction}")

    # Compute evaluation metrics for selective predictions
    sel_metrics, sel_metric_categories = compute_eval_metrics(sel_predictions, 
                                                            sel_targets, 
                                                            sel_uncertainty)
    
    print("Selective Prediction Metrics:")
    for category in sel_metric_categories:
        print(f"{category}: {sel_metrics[category]}")

    # Print all hyperparameters
    print("Hyperparameters:")
    print(f"Memory Depth: {Config.MEMORY_DEPTH}")
    print(f"Cutoff Dimension: {Config.CUTOFF_DIM}")
    print(f"Training Steps: {Config.TRAINING_STEPS}")
    print(f"Learning Rate: {Config.TRAINING_LEARNING_RATE}")


    # Plotting the results
    plot_predictions(
        X_train.numpy(), y_train.numpy(), X_test.numpy(), y_test.numpy(),
        predictions, pred_std=predictive_uncertainty, epistemic=predictive_uncertainty,
        aleatoric=None, title="Memristor Model Predictions vs Targets"
    )

    # # Train and predict with MLP baseline
    # mlp_model = train_mlp_baseline(X_train, y_train, hidden_layers=MLP_HIDDEN_LAYERS, epochs=MLP_EPOCHS, learning_rate=MLP_LEARNING_RATE)
    # mlp_predictions = predict_mlp_baseline(mlp_model, X_test)
    # plot_predictions(
    #     X_train.numpy(), y_train.numpy(), X_test.numpy(), y_test.numpy(),
    #     mlp_predictions, title="MLP Baseline Predictions"
    # )

    # # Train and predict with Polynomial baseline
    # poly_coeffs = train_polynomial_baseline(X_train, y_train, degree=POLYNOMIAL_DEGREE)
    # poly_predictions = predict_polynomial_baseline(poly_coeffs, X_test)
    # plot_predictions(
    #     X_train.numpy(), y_train.numpy(), X_test.numpy(), y_test.numpy(),
    #     poly_predictions, title="Polynomial Baseline Predictions"
    # )

    # # Plot all predictions together
    # plot_all_predictions(X_train, y_train, X_test, y_test, predictions, mlp_predictions, poly_predictions, predictive_uncertainty)

if __name__ == "__main__":
    main()
