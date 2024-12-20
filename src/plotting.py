import matplotlib.pyplot as plt
import os
import matplotlib.pyplot as plt
import numpy as np

# src/plotting.py
# src/plotting.py

def plot_selective_predictions(predictions, targets, uncertainties, remaining_fraction, save_path=None, plot_title='Selective Prediction Results'):
    """Plot selective prediction results with uncertainty bands.
    
    Args:
        predictions: Selected predictions after thresholding
        targets: Selected true targets after thresholding
        uncertainties: Selected uncertainties after thresholding
        remaining_fraction: Fraction of data points retained
        save_path: Path to save the plot (str)
    """
    plt.figure(figsize=(12, 6))
    
    # Convert to numpy arrays if needed
    predictions = predictions.numpy() if hasattr(predictions, 'numpy') else np.array(predictions)
    targets = targets.numpy() if hasattr(targets, 'numpy') else np.array(targets)
    
    # Create indices for x-axis
    x_points = np.arange(len(predictions))
    
    # Sort by predictions for cleaner visualization
    sort_idx = np.argsort(predictions.ravel())
    pred_sorted = predictions.ravel()[sort_idx]
    targets_sorted = targets.ravel()[sort_idx]
    
    # Plot data points
    plt.scatter(x_points, targets_sorted, c='blue', label='True', alpha=0.7, s=30)
    plt.scatter(x_points, pred_sorted, c='red', label='Predicted', alpha=0.7, s=30)
    plt.plot(x_points, pred_sorted, 'r-', alpha=0.3)
    
    # Plot uncertainty bands
    if uncertainties is not None and len(uncertainties) > 0:
        uncert_sorted = uncertainties.ravel()[sort_idx]
        plt.fill_between(
            x_points,
            pred_sorted - 2*uncert_sorted,
            pred_sorted + 2*uncert_sorted,
            color='red', alpha=0.2, label='±2σ'
        )
    
    plt.xlabel('Sample Index')
    plt.ylabel('Value')
    plt.title(f'{plot_title}\nRetained Fraction: {remaining_fraction:.2%}')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
        
def plot_predictions_new(X_test, y_test, predictions, uncertainties=None, save_path=None, plot_title='Quantum Neural Network Predictions'):
    """Plot prediction results with uncertainty bands.
    
    Args:
        X_test: Input test data (tensorflow tensor or numpy array)
        y_test: True test targets (tensorflow tensor or numpy array)
        predictions: Mean predictions (tensorflow tensor or numpy array)
        uncertainties: Standard deviation of predictions (numpy array, optional)
        save_path: Path to save the plot (str)
    """
    plt.figure(figsize=(12, 6))
    
    # Convert tensors to numpy arrays
    X_test = X_test.numpy() if hasattr(X_test, 'numpy') else np.array(X_test)
    y_test = y_test.numpy() if hasattr(y_test, 'numpy') else np.array(y_test)
    predictions = predictions.numpy() if hasattr(predictions, 'numpy') else np.array(predictions)
    
    # Sort all arrays by X values
    sort_idx = np.argsort(X_test.ravel())
    X_sorted = X_test.ravel()[sort_idx]
    y_sorted = y_test.ravel()[sort_idx]
    pred_sorted = predictions.ravel()[sort_idx]
    
    # Plot true values and mean predictions
    plt.scatter(X_sorted, y_sorted, c='blue', label='True', alpha=0.7, s=30)
    plt.scatter(X_sorted, pred_sorted, c='red', label='Predicted (mean)', alpha=0.7, s=30)
    plt.plot(X_sorted, pred_sorted, 'r-', alpha=0.3)
    
    # Plot uncertainty bands if available
    if uncertainties is not None and len(uncertainties) > 0:
        uncert_sorted = uncertainties.ravel()[sort_idx]
        plt.fill_between(
            X_sorted,
            pred_sorted - 2*uncert_sorted,
            pred_sorted + 2*uncert_sorted,
            color='red', alpha=0.2, label='±2σ'
        )
    
    plt.xlabel('Input')
    plt.ylabel('RMSE')
    plt.title(plot_title)
    plt.legend()
    plt.grid(True)
    
    if save_path is not None:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    else:
        plt.show()
    

def plot_eval_metrics(metrics, metric_categories, save_path=None):
    """Plot evaluation metrics from compute_eval_metrics output.
    
    Args:
        metrics: Dictionary of metrics by category
        metric_categories: List of metric categories
        save_path: Path to save the plot
    """
    n_categories = len(metric_categories)
    fig, axes = plt.subplots(1, n_categories, figsize=(6*n_categories, 5))
    
    # Handle single category case
    if n_categories == 1:
        axes = [axes]
    
    for ax, category in zip(axes, metric_categories):
        if category in metrics:
            category_metrics = metrics[category]
            
            # Sort metrics by name
            metric_names = sorted(category_metrics.keys())
            metric_values = [category_metrics[name] for name in metric_names]
            
            # Create bar plot
            bars = ax.bar(range(len(metric_names)), metric_values)
            
            # Customize appearance
            ax.set_xticks(range(len(metric_names)))
            ax.set_xticklabels(metric_names, rotation=45, ha='right')
            ax.set_title(category.replace('_', ' ').title())
            ax.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom')
    
    # Add overall title
    if 'accuracy' in metrics:
        rmse = metrics['accuracy'].get('rmse', 'N/A')
        r2 = metrics['accuracy'].get('r2', 'N/A')
        plt.suptitle(f'Evaluation Metrics (RMSE: {rmse:.3f}, R²: {r2:.3f})')
    
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

    return fig

def plot_training_results(res_mem, filepath=None):
    """
    Plots and saves the training results including loss and parameters over iterations.
    
    Args:
        res_mem: Dictionary containing training history with keys ('loss', 'tr', step)
        filepath: Path where to save the plot
    """
    steps = [k[2] for k in res_mem.keys()]
    losses = [v[0] for v in res_mem.values()]
    phase1_values = [v[1] for v in res_mem.values()]
    phase3_values = [v[2] for v in res_mem.values()]
    weights = [v[3] for v in res_mem.values()]

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    
    ax1.plot(steps, losses, 'b-', zorder=1)  # Line underneath
    ax1.scatter(steps, losses, c='blue', s=30, zorder=2)  # Points on top
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Loss')
    ax1.grid(True)
    
   # Phase 1 plot
    ax2.plot(steps, phase1_values, 'r-', zorder=1)
    ax2.scatter(steps, phase1_values, c='red', s=30, zorder=2)
    ax2.set_title('Phase 1')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Phase')
    ax2.grid(True)

    # Phase 3 plot
    ax3.plot(steps, phase3_values, 'g-', zorder=1)
    ax3.scatter(steps, phase3_values, c='green', s=30, zorder=2)
    ax3.set_title('Phase 3')
    ax3.set_xlabel('Step')
    ax3.set_ylabel('Phase')
    ax3.grid(True)

    # Memristor Weight plot
    ax4.plot(steps, weights, 'm-', zorder=1)
    ax4.scatter(steps, weights, c='magenta', s=30, zorder=2)
    ax4.set_title('Memristor Weight')
    ax4.set_xlabel('Step')
    ax4.set_ylabel('Phase')
    ax4.grid(True)
    
    plt.tight_layout()
    
    if filepath is not None:
        plt.savefig(filepath)
        plt.close(fig)
    else:
        plt.show()

    

def plot_toy_data(X_train, 
                  y_train, 
                  X_test, 
                  y_test):
    """Plot the toy data."""
    fig, ax = plt.subplots(1)
    ax.scatter(X_train, y_train, color="blue", label="train_data")
    ax.scatter(X_test, y_test, color="orange", label="test_data")
    plt.legend()
    plt.show()

def plot_predictions(X_train, 
                     y_train, 
                     X_test, 
                     y_test, 
                     y_pred, 
                     pred_std=None, 
                     pred_quantiles=None, 
                     epistemic=None, 
                     aleatoric=None, 
                     title=None,
                     save_path=None) -> None:
    """Plot predictive uncertainty as well as epistemic and aleatoric separately.
    
    Args:
      X_train: Training input data.
      y_train: Training target data.
      X_test: Test input data.
      y_test: Test target data.
      y_pred: Predicted values.
      pred_std: Standard deviation of predictions (predictive uncertainty).
      pred_quantiles: Quantiles of predictions.
      epistemic: Epistemic uncertainty (for us this is predictive_uncertainty).
      aleatoric: Aleatoric uncertainty.
      title: Title for the plot.
    """
    fig = plt.figure(figsize=(12, 6))
    ax0 = fig.add_subplot(1, 2, 1)

    # Model predictive uncertainty bands on the left
    ax0.scatter(X_test, y_test, color="gray", label="Ground Truth", s=0.5)
    ax0.scatter(X_train, y_train, color="blue", label="Train Data")
    ax0.scatter(X_test, y_pred, color="orange", label="Predictions")

    if pred_std is not None:
        ax0.fill_between(
            X_test.squeeze(),
            y_pred - pred_std,
            y_pred + pred_std,
            alpha=0.3,
            color="tab:red",
            label="$\sqrt{\mathbb{V}\,[y]}$",
        )

    if pred_quantiles is not None:
        ax0.plot(X_test, pred_quantiles, color="tab:red", linestyle="--", label="Quantiles")

    if title is not None:
        ax0.set_title(title + " showing mean ± std")

    ax0.legend()

    # Epistemic and aleatoric uncertainty plots on the right
    ax1 = fig.add_subplot(2, 2, 2)
    if epistemic is not None:
        ax1.plot(X_test, epistemic, color="tab:blue", label="Epistemic Uncertainty")
        ax1.fill_between(
            X_test.squeeze(),
            epistemic - pred_std,
            epistemic + pred_std,
            alpha=0.3,
            color="tab:blue",
        )
        ax1.set_title("Epistemic Uncertainty")
        ax1.legend()

    ax2 = fig.add_subplot(2, 2, 4)
    if aleatoric is not None:
        ax2.plot(X_test, aleatoric, color="tab:green", label="Aleatoric Uncertainty")
        ax2.fill_between(
            X_test.squeeze(),
            aleatoric - pred_std,
            aleatoric + pred_std,
            alpha=0.3,
            color="tab:green",
        )
        ax2.set_title("Aleatoric Uncertainty")
        ax2.legend()

    plt.tight_layout()
    if save_path is not None:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_all_predictions(X_train, y_train, X_test, y_test, memristor_predictions, mlp_predictions, poly_predictions, predictive_uncertainty):
    """Plot all predictions together."""
    plt.figure(figsize=(10, 6))
    plt.scatter(X_train.numpy(), y_train.numpy(), label='Training Data', color='blue')
    plt.scatter(X_test.numpy(), y_test.numpy(), label='Test Data', color='green')
    plt.plot(X_test.numpy(), memristor_predictions, label='Memristor Predictions', color='red')
    plt.plot(X_test.numpy(), mlp_predictions, label='MLP Predictions', color='orange')
    plt.plot(X_test.numpy(), poly_predictions, label='Polynomial Predictions', color='purple')
    plt.fill_between(X_test.numpy().flatten(), 
                     (memristor_predictions - predictive_uncertainty).flatten(), 
                     (memristor_predictions + predictive_uncertainty).flatten(), 
                     color='red', alpha=0.2, label='Predictive Uncertainty')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('Model Predictions vs Targets')
    plt.legend()
    plt.show()

def plot_mlp_architecture_vs_rmse(architectures, rmse_values, save_path=None):
    plt.figure(figsize=(10, 6))
    plt.plot([str(arch) for arch in architectures], rmse_values, marker='o')
    plt.xlabel('MLP Architecture (Hidden Layers)')
    plt.ylabel('RMSE')
    plt.title('MLP Architecture vs RMSE')
    plt.grid(True)
    
    if save_path is not None:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_data(X_train, y_train, X_test, y_test, save_path=None):
    """
    Plot the training and testing data.

    Args:
        X_train: Training input data.
        y_train: Training target data.
        X_test: Testing input data.
        y_test: Testing target data.
        save_path: Optional path to save the plot.
    """
    plt.figure(figsize=(10, 6))
    
    # Plot training data
    plt.scatter(X_train, y_train, color='blue', label='Training Data', alpha=0.6)
    
    # Plot testing data
    plt.scatter(X_test, y_test, color='red', label='Testing Data', alpha=0.6)
    
    plt.xlabel('Input')
    plt.ylabel('Output')
    plt.title('Training and Testing Data')
    plt.legend()
    plt.grid(True)
    
    if save_path is not None:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()



