import matplotlib.pyplot as plt


def plot_toy_data(X_train, y_train, X_test, y_test):
    """Plot the toy data."""
    fig, ax = plt.subplots(1)
    ax.scatter(X_train, y_train, color="blue", label="train_data")
    ax.scatter(X_test, y_test, color="orange", label="test_data")
    plt.legend()
    plt.show()

def plot_predictions(
    X_train, y_train, X_test, y_test, y_pred, pred_std=None, pred_quantiles=None, epistemic=None, aleatoric=None, title=None
) -> None:
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
    plt.show()

