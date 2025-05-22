import numpy as np
from typing import Any, List
from utils.helpers import save_model_params

class Trainer:
    """
    Trainer class for coordinating data, circuit, simulation, autograd, and loss modules.
    """
    def __init__(self, data_source, circuit, simulation_runner, grad_method, loss_fn, config):
        self.data_source = data_source
        self.circuit = circuit
        self.simulation_runner = simulation_runner
        self.grad_method = grad_method
        self.loss_fn = loss_fn
        self.config = config

    def run(self) -> List[float]:
        """
        Run the training loop.
        Returns:
            List[float]: History of loss values
        """
        X_train, y_train, X_test, y_test = self.data_source.get_data()
        enc_train = self.data_source.encode_phase(X_train)
        params = np.array([1.0, 2.0, 0.5])  # Example initial params
        lr = self.config.get('lr', 0.03)
        epochs = self.config.get('epochs', 10)
        loss_history = []
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
                transient=False
            ) as progress:
                task = progress.add_task("Training", total=epochs, loss=0.0)
                for epoch in range(epochs):
                    preds = self.simulation_runner.run_sequence(params, enc_train, self.config['n_samples'])
                    loss = self.loss_fn.compute(preds, y_train)
                    grads = self.grad_method.compute_gradients(params, enc_train, y_train)
                    params -= lr * grads
                    loss_history.append(loss)
                    progress.update(task, advance=1, loss=loss)
            console.print(f"[bold green]Training complete. Final params:[/bold green] {params}")
        except ImportError:
            for epoch in range(epochs):
                preds = self.simulation_runner.run_sequence(params, enc_train, self.config['n_samples'])
                loss = self.loss_fn.compute(preds, y_train)
                grads = self.grad_method.compute_gradients(params, enc_train, y_train)
                params -= lr * grads
                loss_history.append(loss)
                print(f"Epoch {epoch+1}/{epochs} | Loss: {loss:.6f}")
            print("Training complete. Final params:", params)
        # Save final parameters
        param_save_path = self.config.get('param_save_path', 'model_params.npy')
        save_model_params(params, param_save_path)
        return loss_history, params 