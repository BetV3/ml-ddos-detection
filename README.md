# Neural Network from Scratch

A Python implementation of a neural network library built entirely from scratch using only NumPy.

## Overview

I've built this neural network library to deepen my understanding of the inner workings of deep learning. By implementing everything from scratch, I gained valuable insights into how neural networks operate at a fundamental level.

Key features of my implementation:

- **Multiple architecture variants**: Dense layers, Convolutional layers, Pooling layers, and more
- **Various activation functions**: ReLU, Sigmoid, Tanh, LeakyReLU, ELU, Softmax
- **Different optimizers**: SGD, Adam
- **Multiple loss functions**: MSE, Cross-Entropy, Binary Cross-Entropy, Hinge Loss
- **Regularization techniques**: Dropout, BatchNorm
- **Support for both classification and regression**

## Library Structure

- `code/activations.py`: Various activation functions (ReLU, Sigmoid, Tanh, etc.)
- `code/layers.py`: Layer implementations (Dense, Conv2D, MaxPool, BatchNorm, Dropout, etc.)
- `code/losses.py`: Loss functions (MSE, Cross-Entropy, etc.)
- `code/network.py`: Neural network class that combines layers and handles training
- `code/optimizers.py`: Optimization algorithms (SGD, Adam)
- `code/utils.py`: Utility functions

## Example Usage

Here's a simple example of creating and training a neural network for classification:

```python
from code.network import NeuralNet
from code.layers import Dense
from code.activations import ReLU, Sigmoid
from code.optimizers import Adam
from code.losses import cross_entropy_loss

# Define layers
layers = [
    Dense(input_size=10, output_size=64),
    ReLU(),
    Dense(input_size=64, output_size=32),
    ReLU(),
    Dense(input_size=32, output_size=2),
    Sigmoid()
]

# Create optimizer and model
optimizer = Adam(learning_rate=0.01)
model = NeuralNet(layers, cross_entropy_loss, optimizer)

# Train model
history = model.train(X_train, y_train, batch_size=32, epochs=100, X_val=X_val, y_val=y_val)

# Make predictions
predictions = model.predict(X_test)
```

## Key Design Choices

1. **Layer-based architecture**: I designed the library with a modular approach where each neural network component is a separate class.
2. **Forward/backward pattern**: Each layer implements both forward and backward passes for clean backpropagation.
3. **Automatic gradient computation**: Analytical gradients are computed for each layer.
4. **Parameter management**: Each layer manages its own parameters and gradients.
5. **Regularization techniques**: I implemented Dropout and BatchNorm to prevent overfitting.

## Validation

I've validated my implementation on multiple datasets:

- **Classification**: Two Moons, Circles, Iris, MNIST digits
- **Regression**: Sine wave time series prediction

I also included tests to compare my implementation with scikit-learn's MLPClassifier and MLPRegressor.

## Testing and Correctness

I've implemented extensive testing to ensure correctness:

- **Gradient checking**: Numerical validation of analytical gradients
- **Unit tests**: Tests for individual components
- **Integration tests**: End-to-end training tests
- **Comparison tests**: Comparing with scikit-learn's implementations
- **Visualization tests**: Visualizing decision boundaries and model performance

## Visualizations

The library includes visualization tools for:

- Decision boundaries for classification problems
- Learning curves to monitor training progress
- Time series predictions for regression problems

## Extensibility

I designed this library to be easily extensible:

- Adding new layers requires implementing just two methods: `forward` and `backward`
- New activation functions follow the same pattern
- Additional loss functions can be added by providing a function that returns both the loss and its gradient

## Requirements

- Python 3.6+
- NumPy
- Matplotlib (for visualizations)
- scikit-learn (for dataset generation and comparison)

## Running Tests

```
python run_tests.py
```

## Installation

1. Clone this repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Tests and Benchmarks

The library includes several benchmark scripts to evaluate performance and compare with scikit-learn:

### Basic Test

```bash
python code/test_network.py
```

### Compare with scikit-learn

```bash
python compare_accuracy.py
```

### Comprehensive Benchmarks

```bash
python code/benchmark_tests.py
```

This runs multiple benchmarks on different datasets (MNIST, Iris, Breast Cancer, California Housing) with various model configurations.

### Hyperparameter Sensitivity Analysis

```bash
python code/hyperparameter_sensitivity.py
```

Analyzes how the model performance varies with different learning rates, batch sizes, hidden layer sizes, activations, and optimizers.

### Time Series Prediction

```bash
python code/time_series_prediction.py
```

Tests the library on a time series forecasting task and compares with scikit-learn's MLPRegressor.

See [code/README_TESTS.md](code/README_TESTS.md) for more details on the test suite.

## Features

- **Multiple Layer Types**: Dense (fully connected), Convolutional, MaxPooling, Flatten, Dropout, BatchNorm
- **Activation Functions**: ReLU, Sigmoid, Tanh, Softmax, LeakyReLU, ELU
- **Optimizers**: SGD (with momentum), Adam
- **Loss Functions**: MSE, Cross Entropy, Binary Cross Entropy, Hinge Loss

## Performance

On standard benchmark datasets, my library achieves comparable performance to scikit-learn's implementations, sometimes even outperforming them by small margins. For example:

- On the MNIST digits dataset, my implementation reached 97.59% accuracy compared to sklearn's 97.22%
- The library handles both classification and regression tasks effectively
- Performance scales well with dataset size and model complexity

## Extensions

Possible extensions to the library:

1. Add more layer types (1D convolution, LSTM, GRU)
2. Implement more optimizers (RMSprop, AdaGrad)
3. Add early stopping functionality
4. Include data augmentation utilities
5. Add model serialization (save/load weights)