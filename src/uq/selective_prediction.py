import numpy as np
import uncertainty_toolbox as uct

def selective_prediction(final_predictions, targets, predictive_uncertainty, threshold: float = 0.8):
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

