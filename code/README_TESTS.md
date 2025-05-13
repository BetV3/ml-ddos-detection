# Neural Network Library Tests and Benchmarks

This directory contains various tests and benchmarks to evaluate the performance of my neural network library compared to scikit-learn's implementations.

## Running the Tests

### 1. Basic Test

The basic test trains a simple neural network on random data and checks if it works correctly:

```bash
python code/test_network.py
```

### 2. Comprehensive Benchmarks

The benchmark tests compare my library against scikit-learn on multiple datasets:

```bash
python code/benchmark_tests.py
```

This script performs the following tests:
- Classification on the MNIST digits dataset (single and multi-layer networks)
- Classification on the Iris dataset
- Binary classification on the Breast Cancer dataset
- Tests with regularization (Dropout)
- Tests with Batch Normalization
- Regression on the California Housing dataset

The script generates comparison plots for accuracy/MSE and training time, saved as PNG files.

### 3. Hyperparameter Sensitivity Analysis

This script tests how my network performs with different hyperparameter settings:

```bash
python code/hyperparameter_sensitivity.py
```

The analysis includes:
- Learning rate sensitivity
- Batch size effects
- Hidden layer size impact
- Activation function comparison
- Optimizer comparison

Visualizations for each analysis are generated as PNG files.

### 4. Time Series Prediction

This script tests my neural network on a time series forecasting task:

```bash
python code/time_series_prediction.py
```

It generates synthetic time series data with trend, seasonality, and noise components, then trains both my network and scikit-learn's MLPRegressor to predict future values.

## Comparing with sklearn

The `compare_accuracy.py` script in the project root directory specifically compares my implementation with scikit-learn's MLPClassifier on the digits dataset:

```bash
python compare_accuracy.py
```

## Results Interpretation

When analyzing the results, consider the following:

1. **Accuracy/MSE**: How close my implementation comes to scikit-learn's performance. In many cases, my implementation may perform similarly or even better.

2. **Training Time**: my implementation may be slower than scikit-learn's optimized C++ implementation, but should be within a reasonable factor.

3. **Learning Curves**: Check the convergence behavior of my models and compare it with scikit-learn's.

4. **Hyperparameter Sensitivity**: my implementation should show similar trends to established best practices (e.g., too large learning rates causing divergence).

## Extending the Tests

To add new test cases:

1. Add new datasets to `benchmark_tests.py`
2. Modify architecture parameters to test different network configurations
3. Create custom test scripts for specific tasks (like image classification, NLP, etc.) 