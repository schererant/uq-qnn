import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import config
from datetime import datetime
from src.model.mlp import train_mlp_baseline, predict_mlp_baseline
import numpy as np
from src.uq.compute_eval_metrics import compute_eval_metrics
from src.uq.selective_prediction import selective_prediction
from src.model.train import train_memristor
from src.model.predict import predict_memristor
from src.utils.formatting import format_metrics, format_hyperparameters



def hyperparameter_optimization(X_train, y_train, X_test, y_test):
    """
    Performs hyperparameter optimization and stores results in a consistent format
    with model_comparison function, including metrics for all results.
    """

    # log_file_name = Config.LOG_PATH + Config.LOG_FILE_NAME

    if Config.PREDICT_STOCHASTIC:
        Config.PREDICT_SAMPLES = 1

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
            stochastic=Config.PREDICT_STOCHASTIC, 
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
        
        
        if Config.PREDICT_STOCHASTIC:
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