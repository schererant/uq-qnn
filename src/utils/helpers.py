import numpy as np
import matplotlib.pyplot as plt

def set_seed(seed):
    import numpy as np
    import torch
    import random
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

def plot_training_curve(loss_history):
    plt.figure(figsize=(8, 4))
    plt.plot(loss_history, marker='o', color='tab:blue')
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.grid(True, which='both', ls='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

def plot_loss_ascii(loss_history):
    try:
        from rich.console import Console
        from rich.plot import Plot, Series
        console = Console()
        plot = Plot(width=60, height=15)
        plot.add_series(Series(loss_history, label="Loss"))
        console.print("[bold cyan]Loss Curve (log scale):[/bold cyan]")
        console.print(plot)
    except ImportError:
        # fallback: ASCII line plot (not bar plot)
        print("Loss Curve (log scale, ASCII line plot):")
        loss_history = np.array(loss_history)
        epochs = len(loss_history)
        min_loss = np.min(loss_history)
        max_loss = np.max(loss_history)
        log_losses = np.log10(loss_history + 1e-12)
        min_log = np.min(log_losses)
        max_log = np.max(log_losses)
        height = 12
        width = min(epochs, 60)
        # Downsample if too many epochs
        if epochs > width:
            idxs = np.linspace(0, epochs-1, width).astype(int)
            plot_losses = log_losses[idxs]
        else:
            plot_losses = log_losses
            idxs = np.arange(epochs)
        # Normalize to plot height
        y_scaled = (plot_losses - min_log) / (max_log - min_log + 1e-8) * (height-1)
        y_scaled = height - 1 - y_scaled  # invert for plot
        y_scaled = y_scaled.astype(int)
        # Create empty plot
        plot = [[' ' for _ in range(width)] for _ in range(height)]
        # Draw line using simple interpolation
        for i in range(width-1):
            x0, y0 = i, y_scaled[i]
            x1, y1 = i+1, y_scaled[i+1]
            dx = x1 - x0
            dy = y1 - y0
            steps = max(abs(dx), abs(dy))
            for s in range(steps+1):
                x = int(round(x0 + s * dx / steps))
                y = int(round(y0 + s * dy / steps))
                if 0 <= x < width and 0 <= y < height:
                    plot[y][x] = '*' if plot[y][x] == ' ' else plot[y][x]
        # Add y-ticks
        y_tick_labels = np.logspace(max_log, min_log, num=height, base=10)
        y_tick_labels = [f"{10**(max_log - i*(max_log-min_log)/(height-1)):8.2e}" if i % 3 == 0 or i == 0 or i == height-1 else "        " for i in range(height)]
        for i, (row, label) in enumerate(zip(plot, y_tick_labels)):
            print(f"{label} | {''.join(row)}")
        # Add x-ticks
        x_axis = ['-' for _ in range(width)]
        print("         +" + ''.join(x_axis))
        x_labels = [' ']*width
        for i in range(0, width, max(1, width//6)):
            epoch_num = idxs[i]+1
            label = str(epoch_num)
            for j, c in enumerate(label):
                if i+j < width:
                    x_labels[i+j] = c
        print("         |" + ''.join(x_labels))
        print("         | Epochs")
        # Print min/max loss
        print(f"Min loss: {10**min_log:.2e} | Max loss: {10**max_log:.2e}")

def pretty_progress(epoch, epochs, loss):
    try:
        from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn
        from rich.console import Console
        console = Console()
        with Progress(
            TextColumn("Epoch {task.completed}/{task.total}"),
            BarColumn(),
            TextColumn("Loss: {task.fields[loss]:.6f}"),
            TimeElapsedColumn(),
            console=console,
            transient=True
        ) as progress:
            task = progress.add_task("Training", total=epochs, loss=loss)
            progress.update(task, completed=epoch+1, loss=loss)
    except ImportError:
        print(f"Epoch {epoch+1}/{epochs} | Loss: {loss:.6f}")

def plot_perceval_style(X_train, y_train, X_test, y_test, preds, history, enc_swipe=None, X_swipe=None, config=None):
    """
    Reproduce the Perceval-style multi-panel matplotlib plots.
    """
    # Plot 1: loss curve
    plt.figure(figsize=(9, 4))
    plt.plot(history)
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid()

    # Plot 2: data cloud + model fit
    plt.figure(figsize=(9, 5))
    plt.scatter(X_train, y_train, s=20, label='Original data', zorder=3)
    if enc_swipe is not None and X_swipe is not None and config is not None:
        plt.scatter(X_swipe, np.repeat(y_train, config['n_swipe']), s=8, alpha=0.35, label=f'Swipe (n={config["n_swipe"]})', zorder=2)
    plt.plot(X_test, y_test, label='Quartic', ls='--', zorder=1)
    plt.plot(X_test, preds, label='Model', c='red', zorder=4)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid()
    plt.legend()
    plt.tight_layout()

    # Plot 3: index vs encoded phase
    plt.figure(figsize=(9, 4))
    idx = np.arange(len(X_train))
    enc_orig = 2 * np.arccos(X_train)
    plt.plot(idx, enc_orig, '-o', label='Original enc φ', lw=1.5)
    if enc_swipe is not None and config is not None:
        idx_swipe = np.repeat(idx, config['n_swipe'])
        plt.scatter(idx_swipe, enc_swipe, s=6, alpha=0.35, label='Swipe enc φ')
    plt.xlabel('Data index')
    plt.ylabel('Encoding phase [rad]')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()

def save_model_params(params: np.ndarray, filepath: str):
    """
    Save model parameters to a file (numpy .npy format).
    Args:
        params (np.ndarray): Model parameters to save
        filepath (str): Path to save the parameters
    """
    np.save(filepath, params)

def load_model_params(filepath: str) -> np.ndarray:
    """
    Load model parameters from a file (numpy .npy format).
    Args:
        filepath (str): Path to load the parameters from
    Returns:
        np.ndarray: Loaded model parameters
    """
    return np.load(filepath) 