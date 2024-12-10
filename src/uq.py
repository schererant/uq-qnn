import numpy as np
import uncertainty_toolbox as uct
from src.plotting import plot_eval_metrics
import os
from src.utils import format_metrics



def selective_prediction(final_predictions, targets, predictive_uncertainty, threshold: float = 0.8, log_filepath: str = None):
    """ Copmutes UQ metrics
    
    Interpretation:
    - final_predictions: predictions from circuit, np.array
    - targets: data points, np.array
    - predictive_uncertainty: predictive uncertainty from ciruit, np.array 
    - threshold: threshold based on quantiles

    Returns:
     - final_predictions_sel: predictions from circuit, np.array
    - targets_sel: data points, np.array
    - predictive_uncertainty_sel: predictive uncertainty from ciruit, np.array
    - remaining_fraction: remaining data points after selective prediction
    """

    # Convert to numpy arrays
    final_predictions = np.array(final_predictions)
    targets = np.array(targets)
    predictive_uncertainty = np.array(predictive_uncertainty)
    

    predictive_uncertainty_quantile = np.quantile(predictive_uncertainty, threshold)
    results_selected_mask = (predictive_uncertainty < predictive_uncertainty_quantile)

    # hopefully the mask works here
    final_predictions_sel = final_predictions[results_selected_mask]
    targets_sel = targets[results_selected_mask]
    predictive_uncertainty_sel = predictive_uncertainty[results_selected_mask]

    remaining_fraction = len(final_predictions_sel)/len(final_predictions)

    return final_predictions_sel, targets_sel, predictive_uncertainty_sel, remaining_fraction


def compute_eval_metrics(final_predictions,
                         targets, 
                         predictive_uncertainty,
                         logger,
                         param_id
                         ):
    #idea compute eval metrics for selective prediction and full version
    """ Copmutes UQ metrics
    
    Interpretation:
    - final_predictions: predictions from circuit, np.array
    - targets: data points, np.array
    - predictive_uncertainty: predictive uncertainty from ciruit, np.array 

    Returns:
        Dictionary containing all metrics. Accuracy metrics:  Mean average error ('mae'), Root mean squared
        error ('rmse'), Median absolute error ('mdae'),  Mean absolute
        relative percent difference ('marpd'), r^2 ('r2'), and Pearson's
        correlation coefficient ('corr').
    """

    # Convert to numpy arrays
    final_predictions = np.array(final_predictions)
    targets = np.array(targets)
    predictive_uncertainty = np.array(predictive_uncertainty)


    if len(predictive_uncertainty) > 0:
    
        if len(final_predictions) > 0:
            uq_metrics = uct.metrics.get_all_metrics(
                final_predictions,
                predictive_uncertainty,
                targets,
                verbose=False,
                )
        else:
            uq_metrics = [] # TODO: define empty result
        # categories when predictive uncertainty is present
        uq_metric_categories = [
            "scoring_rule",
            "avg_calibration",
            "sharpness",
            "accuracy",
        ]

    else:
        # categories when no predictive uncertainty is present
        uq_metric_categories = ["accuracy"]
        uq_metrics = {
            "accuracy": uct.metrics.get_all_accuracy_metrics(
                final_predictions, targets
            )
        }

    logger.log_evaluation_metrics(uq_metrics, param_id)

    return uq_metrics, uq_metric_categories