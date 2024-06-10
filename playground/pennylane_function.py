
import pennylane as qml
import numpy as np
from lightning_uq_box.viz_utils import (
    plot_calibration_uq_toolbox,
    plot_predictions_regression,
    plot_toy_regression_data,
    plot_training_metrics,
)
from lightning_uq_box.datamodules import ToyHeteroscedasticDatamodule

dm = ToyHeteroscedasticDatamodule()

X_train, y_train, train_loader, X_test, y_test, test_loader = (
    dm.X_train,
    dm.y_train,
    dm.train_dataloader(),
    dm.X_test,
    dm.y_test,
    dm.test_dataloader(),
)

fig = plot_toy_regression_data(X_train, y_train, X_test, y_test)