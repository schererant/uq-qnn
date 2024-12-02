import numpy as np
import uncertainty_toolbox as uct

def compute_eval_metrics(final_predictions, targets, predictive_uncertainty):
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

    return uq_metrics, uq_metric_categories