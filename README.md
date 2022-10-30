# EPFL CS-433: PROJECT 1 "Higgs Boson Challenge"

The Higgs boson is an elementary particle which explains why other particles have a mass. Measurements during high-speed collisions of protons at CERN were made public with the aim of predicting whether the collision by-products are an actual boson or background noise.

The work was mainly done in 2 ways: data pre-processing and then applying logistic regression.  

Preprocessing can include different combinations of the following methods: (1) replacing undefined datapoints by the median/mean, (2) performing a polynomial expansion, (3) standardizing.

Logistic Regressions are subsequently implemented and legitimized by means of a 7-fold cross validation.

The entire project only uses python the libraries Numpy and Matplolib (for visualisation). 

## Code description 

### `run.py`

This file produces the predictions same file used to obtain the team's ("no_CS_members") the best score on the aicrowd platform. It is self-contained and only requires access to the data and files described below.

---

### `implementations.py`

This file contains the required functions as stated in the project outline pdf file.

* *mean_squared_error_gd, mean_squared_error_sgd, least_squares, ridge_regression*
* *logistic_regression, reg_logistic_regression*

As well as auxiliary functions supporting the ones cited above.

* *compute_mse_loss, compute_mse_gradient, batch_iter, compute_stoch_mse_gradient, sigmoid, calculate_logistic_loss, calculate_logistic_gradient*
* *calculate_stoch_logistic_gradient, stoch_reg_logistic_regression*

---

### `data_processing.py`

This file contains functions used to pre-process data.

* *data_removed, data_replaced, split_data, add_w0*
* *normalize_log_gaussian, normalize_angles, normalize_gaussian, normalize_min_max, normalize*

--- 

### `hyperparams.py`

* *build_k_indices, cross_validation, cross_validation_demo, build_poly, best_degree_selection, phi_optimized*

--- 

### `our_progress_run.ipynb`

A notebook outlining the step-by-step progress of the model (each stage adds something on top of the previous version):

1. logistic regression 
2. logistic regression + normalized 
3. logistic regression + normalized + w0
4. logistic regression + normalized smart + w0
5. logistic regression + normalized smart + w0 + polynomial expansion w/ Ridge regression

## Authors 

* Mathilde Morini
* Iris Toye
* Alexei Ermochkine
