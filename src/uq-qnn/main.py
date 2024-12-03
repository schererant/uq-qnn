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
from plotting import plot_predictions, plot_training_results, plot_predictions_new, plot_eval_metrics
from baseline import train_mlp_baseline, predict_mlp_baseline, train_polynomial_baseline, predict_polynomial_baseline
from uq import selective_prediction, compute_eval_metrics
from model import train_memristor, predict_memristor, build_circuit
from utils import format_metrics, format_hyperparameters

tf.get_logger().setLevel('ERROR')
warnings.filterwarnings("ignore")

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
rd.seed(42)


###### HYPERPARAMETERS ######

class Config:

    
    LOG_NAME = f"logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    LOG_FILE_NAME = f"reports/logs/experiment_{LOG_NAME}/log.txt"
    LOG_PATH = f"reports/logs/experiment_{LOG_NAME}/"
    
    # Hyperparameter Optimization
    HYPERPARAMETER_OPTIMIZATION = True
    HYPER_STEPS_RANGE = [5, 5, 5]
    HYPER_LEARNING_RATE_RANGE = [0.01]
    HYPER_MEMORY_DEPTH_RANGE = [6]
    HYPER_CUTOFF_DIM_RANGE = [5]

    # Model Comparison
    MODEL_COMPARISON = False
    COMP_N_SAMPLES = [2]
    COMP_MLP_ARCH = [[32], [64, 64], [128, 64, 64]]

    #TODO: Check for which params we have the same loss
    MLP_HIDDEN_LAYERS = [64, 64]
    MLP_EPOCHS = 100
    MLP_LEARNING_RATE = 0.01

    POLYNOMIAL_DEGREE = 3

    # Selective Prediction
    SELECTIVE_PREDICTION_THRESHOLD = 0.8

    # QNN Hyperparameters
    MEMORY_DEPTH = 6
    CUTOFF_DIM = 5
    TRAINING_STEPS = 5
    TRAINING_LEARNING_RATE = 0.01

    PREDICT_STOCHASTIC = False
    PREDICT_SAMPLES = 2
    PREDICT_VARIANCE = 0.1

    GET_DATA_N_DATA = 200
    GET_DATA_SIGMA_NOISE_1 = 0.1
    GET_DATA_DATAFUNCTION = quartic_data

    PARAM_ID = f"qnn_hp_s{TRAINING_STEPS}_lr{TRAINING_LEARNING_RATE}_md{MEMORY_DEPTH}_cd{CUTOFF_DIM}"


def model_comparison(X_train, y_train, X_test, y_test):
    all_results = {}
    
    # 1. MLP Baseline

    print("Training MLP Baseline Models...")
    mlp_architectures = Config.COMP_MLP_ARCH
    
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
        cutoff_dim=Config.CUTOFF_DIM,
        filename=Config.LOG_FILE_NAME
    )
    

    # 3. QNN with UQ
    
    for n_samples in Config.COMP_N_SAMPLES:

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
            var=Config.PREDICT_VARIANCE,
            filename=Config.LOG_FILE_NAME
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
        for threshold in [0.7]:

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


def hyperparameter_optimization(X_train, y_train, X_test, y_test):
    """
    Performs hyperparameter optimization and stores results in a consistent format
    with model_comparison function, including metrics for all results.
    """

    # log_file_name = Config.LOG_PATH + Config.LOG_FILE_NAME

    best_loss = float('inf')
    best_params = None
    best_metrics = None
    best_categories = None
    all_results = {}


    with open(Config.LOG_FILE_NAME, "a") as f:
        f.write("=" * 80 + "\n")
        f.write("Hyperparameter Optimization Log\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")
        f.write("Hyperparameter Ranges:\n")
        f.write(f"Steps Range: {Config.HYPER_STEPS_RANGE}\n")
        f.write(f"Learning Rate Range: {Config.HYPER_LEARNING_RATE_RANGE}\n")
        f.write(f"Memory Depth Range: {Config.HYPER_MEMORY_DEPTH_RANGE}\n")
        f.write(f"Cutoff Dimension Range: {Config.HYPER_CUTOFF_DIM_RANGE}\n\n")
    
    # Create a tqdm iterator with total number of combinations
    for steps, learning_rate, memory_depth, cutoff_dim in tqdm(
        product(Config.HYPER_STEPS_RANGE, Config.HYPER_LEARNING_RATE_RANGE, Config.HYPER_MEMORY_DEPTH_RANGE, Config.HYPER_CUTOFF_DIM_RANGE),
        total=len(Config.HYPER_STEPS_RANGE) * len(Config.HYPER_LEARNING_RATE_RANGE) * len(Config.HYPER_MEMORY_DEPTH_RANGE) * len(Config.HYPER_CUTOFF_DIM_RANGE),
        desc="Hyperparameter Optimization",
        unit="combination"
    ):

    # for steps, learning_rate, memory_depth, cutoff_dim in [(steps_range[0], learning_rate_range[0], memory_depth_range[0], cutoff_dim_range[0])]:
        # print(f"Training with steps={steps}, learning_rate={learning_rate}, memory_depth={memory_depth}, cutoff_dim={cutoff_dim}")

        # Create a unique identifier for this parameter combination

        Config.TRAINING_STEPS = steps
        Config.TRAINING_LEARNING_RATE = learning_rate
        Config.MEMORY_DEPTH = memory_depth
        Config.CUTOFF_DIM = cutoff_dim
        Config.PARAM_ID = f"qnn_hp_s{Config.TRAINING_STEPS}_lr{Config.TRAINING_LEARNING_RATE}_md{Config.MEMORY_DEPTH}_cd{Config.CUTOFF_DIM}"

        # log_dic[param_id] = {
        #     "res_mem": {},
        #     "hyperparameters": {
        #         "steps": steps,
        #         "learning_rate": learning_rate,
        #         "memory_depth": memory_depth,
        #         "cutoff_dim": cutoff_dim
        #     }
        # }

        # Train the memristor model
        res_mem, phase1, phase3, memristor_weight = train_memristor(
            X_train, 
            y_train, 
            memory_depth=memory_depth, 
            training_steps=steps, 
            learning_rate=learning_rate, 
            cutoff_dim=cutoff_dim,
            log_filepath=Config.LOG_FILE_NAME,
            log_path=Config.LOG_PATH,
            param_id=Config.PARAM_ID
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
            samples=Config.PREDICT_SAMPLES, 
            var=Config.PREDICT_VARIANCE, 
            cutoff_dim=Config.CUTOFF_DIM,
            log_filepath=Config.LOG_FILE_NAME,
            log_path=Config.LOG_PATH,
            param_id=Config.PARAM_ID
        )

        
        # Compute evaluation metrics
        metrics, metric_categories = compute_eval_metrics(predictions, 
                                                          targets, 
                                                          predictive_uncertainty,
                                                          Config.LOG_FILE_NAME,
                                                          Config.PARAM_ID
                                                          )

        

        
        # Store results in the same format as model_comparison

        all_results[Config.PARAM_ID] = {
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
            best_hyperparams = (steps, learning_rate, memory_depth, cutoff_dim)
            best_metrics = metrics
            best_categories = metric_categories
            best_params = (phase1, phase3, memristor_weight)


    # Add best parameters summary to results with metrics
    if best_hyperparams:
        all_results["best_parameters"] = {
            "metrics": best_metrics,
            "categories": best_categories,
            "hyperparameters": {
                "steps": best_hyperparams[0],
                "learning_rate": best_hyperparams[1],
                "memory_depth": best_hyperparams[2],
                "cutoff_dim": best_hyperparams[3],
                "best_rmse": best_loss
            },
            "parameters": {
                "phase 1": best_params[0],
                "phase 3": best_params[1],
                "memristor weight": best_params[2]
            }
        }
    else:
        print("No valid hyperparameter combinations found.")

    with open(Config.LOG_FILE_NAME, "a") as f:
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

                if "parameters" in result:
                    f.write("\n\nParameters:\n")
                    parameter_lines = format_metrics(result["parameters"], indent=2)
                    f.write("\n".join(parameter_lines))
                
                f.write("\n\n")

    return best_hyperparams, all_results



def main():

    # Create directory called experiment_CONFIG.LOG_NAME in reports/logs
    os.makedirs(f"reports/logs/experiment_{Config.LOG_NAME}", exist_ok=False)

    with open(Config.LOG_FILE_NAME, "a") as f:
        f.write("=" * 80 + "\n")
        f.write(f"Experiment_{Config.LOG_NAME}\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")


    X_train, y_train, X_test, y_test, _ = get_data(n_data=Config.GET_DATA_N_DATA, 
                                                   sigma_noise_1=Config.GET_DATA_SIGMA_NOISE_1, 
                                                   datafunction=Config.GET_DATA_DATAFUNCTION
                                                   )
    

    if Config.HYPERPARAMETER_OPTIMIZATION:
        # Perform hyperparameter optimization
        best_hyperparams, hpo_results = hyperparameter_optimization(X_train, y_train, X_test, y_test)
        steps, learning_rate, memory_depth, cutoff_dim = best_hyperparams
        # print(hpo_results)


    if Config.MODEL_COMPARISON:
        comp_results = model_comparison(X_train, y_train, X_test, y_test)
        # save_results_to_txt(all_results, f"model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        # print(comp_results)

    
    if not Config.HYPERPARAMETER_OPTIMIZATION and not Config.MODEL_COMPARISON:
        # Train the memristor model
        res_mem, phase1, phase3, memristor_weight = train_memristor(X_train, 
                                                                    y_train, 
                                                                    memory_depth=Config.MEMORY_DEPTH, 
                                                                    training_steps=Config.TRAINING_STEPS,
                                                                    learning_rate=Config.TRAINING_LEARNING_RATE,
                                                                    cutoff_dim=Config.CUTOFF_DIM,
                                                                    log_filepath=Config.LOG_FILE_NAME,
                                                                    log_path=Config.LOG_PATH,
                                                                    param_id=Config.PARAM_ID
                                                                    )

        # Save training results
        # with open(f"{Config.LOG_FILE_NAME}.pkl", "wb") as file:
        #     pickle.dump(res_mem, file)

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
                                                                        cutoff_dim=Config.CUTOFF_DIM,
                                                                        log_filepath=Config.LOG_FILE_NAME,
                                                                        log_path=Config.LOG_PATH,
                                                                        param_id=Config.PARAM_ID
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
                                                                    predictive_uncertainty,
                                                                    Config.LOG_FILE_NAME,
                                                                    Config.PARAM_ID,
                                                                    name="Full Prediction"
                                                                    )
        

        # Apply selective prediction
        sel_predictions, sel_targets, sel_uncertainty, remaining_fraction = selective_prediction(predictions, 
                                                                                                targets, 
                                                                                                predictive_uncertainty, 
                                                                                                threshold=Config.SELECTIVE_PREDICTION_THRESHOLD
                                                                                                )

        # Compute evaluation metrics for selective predictions
        sel_metrics, sel_metric_categories = compute_eval_metrics(sel_predictions, 
                                                                sel_targets, 
                                                                sel_uncertainty,
                                                                Config.LOG_FILE_NAME,
                                                                Config.PARAM_ID,
                                                                name="Selective Prediction"
                                                                )

        # Save results to log file
        with open(Config.LOG_FILE_NAME, "a") as f:
            f.write(f"Selective Prediction Fraction: {remaining_fraction}\n")
            f.write("\n\n")

        # Plotting the results
        plot_predictions(
            X_train.numpy(), y_train.numpy(), X_test.numpy(), y_test.numpy(),
            predictions, pred_std=predictive_uncertainty, epistemic=predictive_uncertainty,
            aleatoric=None, title="Memristor Model Predictions vs Targets", save_path=Config.LOG_PATH+f"prediction_uncertainty_{Config.PARAM_ID}.png"
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
