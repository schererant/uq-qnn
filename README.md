# Uncertainty Quantification with Quantum Neural Networks on Integrated Photonic Circuits
## TODOs:

- initial study of QNNs with UQ in simulation, use 1-d regression function from photonic quantum memristor paper from Iris (ask Iris about simulations)
- use existing QNNs works for regression and classification to check if "inherent Quantum"-UQ adds a benefit:
    - **classification task:** over multiple forward passes compute the mean of logits and then take the softmax, as e.g. in [Link](https://github.com/lightning-uq-box/lightning-uq-box/blob/acd1fc2bfc33860111c272be767e0ddcf1f5b34f/lightning_uq_box/uq_methods/utils.py#L166)
    - compute Entropy over softmax outputs as in [Link](https://github.com/lightning-uq-box/lightning-uq-box/blob/acd1fc2bfc33860111c272be767e0ddcf1f5b34f/lightning_uq_box/uq_methods/utils.py#L186)
    - on a validation set (it should kinda all be iid and similar splits) compute the quantiles of the entropies (per prediction) as in [Pandas Link](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.quantile.html). An example is given in [Link](https://github.com/nilsleh/tropical_cyclone_uq/blob/main/src/class_results_analysis.ipynb) - code box 12, titled "Selective Prediction" 

:::info
**Selective prediction in a nutshell:**  UQ evaluation with selective prediction, as introduced in [Paper](https://proceedings.neurips.cc/paper_files/paper/2017/file/4a8423d5e91fda00bb7e46540e2b0cf1-Paper.pdf). Here, samples with with a predictive uncertainty **(classification: entropies, regression: standard deviation)** above a given threshold are omitted from prediction and referred to an expert and optionally another method. If the corresponding UQ method has higher uncertainties for inaccurate predictions, leaving out the predictions for these samples should increase the overall accuracy. This could resemble a deployment scenario, where predictions are monitored and if the predictive uncertainties surpass a given threshold, the sample is referred to an expert and/or additionally evaluated with another method. Instead of a fixed threshold on the predictive uncertainties across methods, one can chose a UQ specific threshold based on the 0.8 quantile of predictive uncertainties computed on a held out validation dataset for each method. These method-specific thresholds are then utilized on the separate test set for which we report results.

:::

- **regression task:** over multiple forward passes compute the mean of predictions [Link](https://github.com/lightning-uq-box/lightning-uq-box/blob/acd1fc2bfc33860111c272be767e0ddcf1f5b34f/lightning_uq_box/uq_methods/utils.py#L118) and standard deviation, as e.g. in  [Link](https://github.com/lightning-uq-box/lightning-uq-box/blob/acd1fc2bfc33860111c272be767e0ddcf1f5b34f/lightning_uq_box/uq_methods/utils.py#L151)
-  on a validation set (it should kinda all be iid and similar splits) compute the quantiles of the standard deviations (per prediction) as in [Pandas Link](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.quantile.html). An example is given in [Link](https://github.com/nilsleh/tropical_cyclone_uq/blob/main/src/results_tropical.ipynb) - code box 6, titled "selective prediction thresholds based on validation set" 


- compute accuracy according to different quantile UQ thresholds on test datasets, plot should look like:

![](https://s3.desy.de/hackmd/uploads/b40060f6-6324-4a3a-bbfa-5f042bf76474.png)
Above is the RMSE with Selective Prediction Results across Quantiles for the Tropical Cyclone Dataset.

