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
