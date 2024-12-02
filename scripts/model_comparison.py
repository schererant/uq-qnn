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




def model_comparison(X_train, y_train, X_test, y_test):
    all_results = {}

    with open(config.log_file_name, "a") as f:
            # Write header
            f.write("=" * 80 + "\n")
            f.write("Model Comparison Log\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")
    
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
            epochs=config.mlp.epochs,
            learning_rate=config.mlp.learning_rate
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

    with open(config.log_file_name, "a") as f:
        f.write("\nMLP Baseline Models:\n")
        f.write("-" * 40 + "\n")
        for hidden_layers in mlp_architectures:
            f.write(f"Hidden Layers: {hidden_layers}\n")
            f.write("-" * 40 + "\n")
            f.write("\nMetrics:\n")
            metric_lines = format_metrics(all_results[f"mlp_{len(hidden_layers)}_layers"]["metrics"], indent=2)
            f.write("\n".join(metric_lines))
            f.write("\n\n")

    
    # 2. QNN Base Model

    with open(config.log_file_name, "a") as f:
        f.write("QNN Base Model:\n")
        # f.write("-" * 40 + "\n")
    
    print("Training QNN Base Model...")
    res_mem, phase1, phase3, memristor_weight = train_memristor(
        X_train, 
        y_train,
        memory_depth=config.training.memory_depth,
        training_steps=config.training.steps,
        learning_rate=config.training.learning_rate,
        cutoff_dim=config.training.cutoff_dim,
        filename=config.log_file_name
    )

    with open(config.log_file_name, "a") as f:
        f.write("\n")

    # 3. QNN with UQ

    with open(config.log_file_name, 'a') as f:
        f.write("QNN UQ Model:\n")
        f.write("-" * 40 + "\n")
    
    for n_samples in config.model_comparison.n_samples:

        print(f"Predict QNN UQ Model with {n_samples} samples...")
        predictions, targets, predictive_uncertainty = predict_memristor(
            X_test, 
            y_test,
            memory_depth=config.training.memory_depth,
            phase1=phase1,
            phase3=phase3,
            memristor_weight=memristor_weight,
            stochastic=True,
            samples=n_samples,
            var=config.prediction.variance,
            cutoff_dim=config.training.cutoff_dim,
            filename=config.log_file_name
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

        with open(config.log_file_name, "a") as f:
            f.write(f"\nQNN UQ Model with {n_samples} samples:\n")
            f.write("-" * 40 + "\n")
            f.write("\nMetrics:\n")
            metric_lines = format_metrics(full_metrics, indent=2)
            f.write("\n".join(metric_lines))
            f.write("\n\n")

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

            with open(config.log_file_name, "a") as f:
                f.write(f"Selective Prediction with threshold {threshold}:\n")
                f.write("-" * 40 + "\n")
                f.write("\nMetrics:\n")
                metric_lines = format_metrics(sel_metrics, indent=2)
                f.write("\n".join(metric_lines))
                f.write(f"\nRemaining Fraction: {remaining_fraction:.2%}\n\n")

    return all_results


