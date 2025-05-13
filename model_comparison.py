"""
Comprehensive Model Comparison

This script compares my neural network implementation with scikit-learn's 
MLPClassifier on various datasets. I evaluate both implementations on 
classification accuracy, training time, and visualize decision boundaries.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.datasets import make_moons, make_circles, make_classification, load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import from my neural network library
from code.network import NeuralNet
from code.layers import Dense
from code.activations import ReLU, Sigmoid, Tanh
from code.optimizers import Adam
from code.losses import cross_entropy_loss, mse_loss
from code.utils import one_hot_encode

def prepare_dataset(dataset_name, n_samples=500, random_state=42):
    """
    Prepare a dataset for comparison.
    
    Args:
        dataset_name: Name of the dataset to use
        n_samples: Number of samples to generate for synthetic datasets
        random_state: Random seed for reproducibility
        
    Returns:
        X: Features
        y: Labels
        X_train, X_test, y_train, y_test: Train/test split
    """
    if dataset_name == "moons":
        X, y = make_moons(n_samples=n_samples, noise=0.3, random_state=random_state)
    elif dataset_name == "circles":
        X, y = make_circles(n_samples=n_samples, noise=0.2, factor=0.5, random_state=random_state)
    elif dataset_name == "linear":
        X, y = make_classification(n_samples=n_samples, n_features=2, n_redundant=0, 
                                   n_informative=2, random_state=random_state, 
                                   n_clusters_per_class=1)
    elif dataset_name == "iris":
        iris = load_iris()
        X, y = iris.data[:, :2], iris.target  # Using only first two features for visualization
    elif dataset_name == "regression":
        # Create a simple regression dataset (sin function with noise)
        X = np.linspace(-3, 3, n_samples).reshape(-1, 1)
        y = np.sin(X).reshape(-1, 1) + 0.1 * np.random.randn(n_samples, 1)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_state)
    
    return X, y, X_train, X_test, y_train, y_test

def train_my_classification_model(X_train, y_train, num_classes, hidden_size=10, epochs=100):
    """
    Train my neural network implementation for classification.
    
    Args:
        X_train: Training features
        y_train: Training labels
        num_classes: Number of classes
        hidden_size: Size of hidden layers
        epochs: Number of training epochs
        
    Returns:
        model: Trained model
        train_time: Time taken for training (in seconds)
        history: Training history
    """
    # One-hot encode labels
    y_train_one_hot = one_hot_encode(y_train, num_classes)
    
    # Define model architecture
    layers = [
        Dense(X_train.shape[1], hidden_size),
        ReLU(),
        Dense(hidden_size, hidden_size),
        ReLU(),
        Dense(hidden_size, num_classes),
        Sigmoid()
    ]
    
    # Create optimizer and model
    optimizer = Adam(learning_rate=0.01)
    model = NeuralNet(layers, cross_entropy_loss, optimizer)
    
    # Train model and measure time
    start_time = time.time()
    history = model.train(X_train, y_train_one_hot, batch_size=32, epochs=epochs)
    train_time = time.time() - start_time
    
    return model, train_time, history

def train_my_regression_model(X_train, y_train, hidden_size=10, epochs=100):
    """
    Train my neural network implementation for regression.
    
    Args:
        X_train: Training features
        y_train: Training targets
        hidden_size: Size of hidden layers
        epochs: Number of training epochs
        
    Returns:
        model: Trained model
        train_time: Time taken for training (in seconds)
        history: Training history
    """
    # Define model architecture
    layers = [
        Dense(X_train.shape[1], hidden_size),
        ReLU(),
        Dense(hidden_size, hidden_size),
        ReLU(),
        Dense(hidden_size, 1)  # Output: 1 neuron for regression
    ]
    
    # Create optimizer and model
    optimizer = Adam(learning_rate=0.01)
    model = NeuralNet(layers, mse_loss, optimizer)
    
    # Train model and measure time
    start_time = time.time()
    history = model.train(X_train, y_train, batch_size=32, epochs=epochs)
    train_time = time.time() - start_time
    
    return model, train_time, history

def train_sklearn_classification_model(X_train, y_train, hidden_size=10, max_iter=100):
    """
    Train scikit-learn's MLPClassifier.
    
    Args:
        X_train: Training features
        y_train: Training labels
        hidden_size: Size of hidden layers
        max_iter: Maximum number of iterations
        
    Returns:
        model: Trained model
        train_time: Time taken for training (in seconds)
    """
    # Define model
    model = MLPClassifier(hidden_layer_sizes=(hidden_size, hidden_size), 
                         activation='relu', solver='adam', 
                         learning_rate_init=0.01,
                         max_iter=max_iter, random_state=42)
    
    # Train model and measure time
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    return model, train_time

def train_sklearn_regression_model(X_train, y_train, hidden_size=10, max_iter=100):
    """
    Train scikit-learn's MLPRegressor.
    
    Args:
        X_train: Training features
        y_train: Training targets
        hidden_size: Size of hidden layers
        max_iter: Maximum number of iterations
        
    Returns:
        model: Trained model
        train_time: Time taken for training (in seconds)
    """
    # Define model
    model = MLPRegressor(hidden_layer_sizes=(hidden_size, hidden_size), 
                         activation='relu', solver='adam', 
                         learning_rate_init=0.01,
                         max_iter=max_iter, random_state=42)
    
    # Train model and measure time
    start_time = time.time()
    model.fit(X_train, y_train.ravel())
    train_time = time.time() - start_time
    
    return model, train_time

def plot_decision_boundary(ax, model, X, y, is_sklearn=False, title="Decision Boundary"):
    """
    Plot decision boundary for a model.
    
    Args:
        ax: Matplotlib axis object
        model: Trained model
        X: Input features
        y: True labels
        is_sklearn: Whether the model is from scikit-learn
        title: Plot title
    """
    # Set up mesh grid
    h = 0.02  # step size in the mesh
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Create mesh grid points and predict
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    
    if is_sklearn:
        Z = model.predict(mesh_points)
    else:
        # My model
        predictions = model.predict(mesh_points)
        Z = np.argmax(predictions, axis=1)
    
    # Reshape the predictions
    Z = Z.reshape(xx.shape)
    
    # Plot decision boundary
    ax.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.coolwarm)
    
    # Plot class samples
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolor='k', s=40)
    
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_title(title)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')

def plot_regression_results(X, y, my_model, sklearn_model, title="Regression Comparison"):
    """
    Plot regression results comparing my model with scikit-learn.
    
    Args:
        X: Input features
        y: True targets
        my_model: My trained model
        sklearn_model: Trained scikit-learn model
        title: Plot title
    """
    # Sort X for better visualization
    idx = np.argsort(X.flatten())
    X_sorted = X[idx]
    y_sorted = y[idx]
    
    # Get predictions
    my_preds = my_model.predict(X_sorted)
    sklearn_preds = sklearn_model.predict(X_sorted).reshape(-1, 1)
    
    # Create figure
    plt.figure(figsize=(12, 6))
    
    # Plot true data
    plt.scatter(X_sorted, y_sorted, alpha=0.5, label='True values', color='blue')
    
    # Plot predictions
    plt.plot(X_sorted, my_preds, 'r-', linewidth=2, label='My model')
    plt.plot(X_sorted, sklearn_preds, 'g--', linewidth=2, label='sklearn model')
    
    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.savefig('regression_comparison.png')
    print("Regression comparison plot saved as 'regression_comparison.png'")
    
    # Calculate MSE
    my_mse = mean_squared_error(y_sorted, my_preds)
    sklearn_mse = mean_squared_error(y_sorted, sklearn_preds)
    
    print(f"My model MSE: {my_mse:.6f}")
    print(f"sklearn model MSE: {sklearn_mse:.6f}")
    
    return {'my_mse': my_mse, 'sklearn_mse': sklearn_mse}

def compare_classification_models(dataset_name, hidden_size=10, epochs=100):
    """
    Compare my implementation with scikit-learn on a specific classification dataset.
    
    Args:
        dataset_name: Name of the dataset to use
        hidden_size: Size of hidden layers
        epochs: Number of training epochs
        
    Returns:
        Dictionary of comparison results
    """
    print(f"\n=== Comparing on {dataset_name} dataset (Classification) ===")
    
    # Prepare dataset
    X, y, X_train, X_test, y_train, y_test = prepare_dataset(dataset_name)
    num_classes = len(np.unique(y))
    
    print(f"Dataset: {dataset_name}, shape: {X.shape}, classes: {num_classes}")
    
    # Train my model
    print("Training my model...")
    my_model, my_time, history = train_my_classification_model(X_train, y_train, num_classes, 
                                                   hidden_size=hidden_size, epochs=epochs)
    
    # Train scikit-learn model
    print("Training scikit-learn model...")
    sklearn_model, sklearn_time = train_sklearn_classification_model(X_train, y_train, 
                                                      hidden_size=hidden_size, max_iter=epochs)
    
    # Evaluate my model
    my_pred = np.argmax(my_model.predict(X_test), axis=1)
    my_accuracy = accuracy_score(y_test, my_pred)
    
    # Evaluate scikit-learn model
    sklearn_pred = sklearn_model.predict(X_test)
    sklearn_accuracy = accuracy_score(y_test, sklearn_pred)
    
    # Print results
    print(f"My implementation - Accuracy: {my_accuracy:.4f}, Training time: {my_time:.2f}s")
    print(f"Scikit-learn - Accuracy: {sklearn_accuracy:.4f}, Training time: {sklearn_time:.2f}s")
    
    # Create figure for visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot decision boundaries
    plot_decision_boundary(ax1, my_model, X, y, is_sklearn=False, 
                          title=f"My Implementation\nAccuracy: {my_accuracy:.4f}")
    plot_decision_boundary(ax2, sklearn_model, X, y, is_sklearn=True, 
                          title=f"Scikit-learn MLPClassifier\nAccuracy: {sklearn_accuracy:.4f}")
    
    plt.tight_layout()
    plt.savefig(f"{dataset_name}_comparison.png")
    print(f"Decision boundary comparison saved as '{dataset_name}_comparison.png'")
    
    # Return comparison results
    return {
        'dataset': dataset_name,
        'my_accuracy': my_accuracy,
        'sklearn_accuracy': sklearn_accuracy,
        'my_time': my_time,
        'sklearn_time': sklearn_time
    }

def compare_regression_models(hidden_size=10, epochs=100):
    """
    Compare my implementation with scikit-learn on a regression dataset.
    
    Args:
        hidden_size: Size of hidden layers
        epochs: Number of training epochs
        
    Returns:
        Dictionary of comparison results
    """
    print("\n=== Comparing on regression dataset ===")
    
    # Prepare dataset
    X, y, X_train, X_test, y_train, y_test = prepare_dataset("regression")
    
    print(f"Dataset: regression, shape: {X.shape}")
    
    # Train my model
    print("Training my regression model...")
    my_model, my_time, history = train_my_regression_model(X_train, y_train, 
                                           hidden_size=hidden_size, epochs=epochs)
    
    # Train scikit-learn model
    print("Training scikit-learn regression model...")
    sklearn_model, sklearn_time = train_sklearn_regression_model(X_train, y_train, 
                                                hidden_size=hidden_size, max_iter=epochs)
    
    # Plot regression results
    results = plot_regression_results(X, y, my_model, sklearn_model, 
                        title="Regression Comparison: My Model vs. sklearn")
    
    # Calculate MSE on test set
    my_preds = my_model.predict(X_test)
    sklearn_preds = sklearn_model.predict(X_test).reshape(-1, 1)
    
    my_test_mse = mean_squared_error(y_test, my_preds)
    sklearn_test_mse = mean_squared_error(y_test, sklearn_preds)
    
    print(f"Test MSE - My model: {my_test_mse:.6f}, sklearn: {sklearn_test_mse:.6f}")
    
    # Return comparison results
    return {
        'dataset': 'regression',
        'my_mse': my_test_mse,
        'sklearn_mse': sklearn_test_mse,
        'my_time': my_time,
        'sklearn_time': sklearn_time
    }

def run_all_comparisons():
    """
    Run comparisons on all datasets and summarize results.
    """
    classification_datasets = ["moons", "circles", "linear", "iris"]
    classification_results = {}
    
    # Run classification comparisons
    for dataset in classification_datasets:
        classification_results[dataset] = compare_classification_models(dataset, hidden_size=20, epochs=100)
    
    # Run regression comparison
    regression_results = compare_regression_models(hidden_size=20, epochs=100)
    
    # Create summary table for classification
    print("\n=== Summary of Classification Comparisons ===")
    print("| Dataset | My Accuracy | Scikit-learn Accuracy | My Time (s) | Scikit-learn Time (s) |")
    print("|---------|-------------|----------------------|-------------|------------------------|")
    
    for dataset, result in classification_results.items():
        print(f"| {dataset.ljust(7)} | {result['my_accuracy']:.4f} | {result['sklearn_accuracy']:.4f} | {result['my_time']:.2f} | {result['sklearn_time']:.2f} |")
    
    # Create summary for regression
    print("\n=== Summary of Regression Comparison ===")
    print("| Dataset    | My MSE   | Scikit-learn MSE | My Time (s) | Scikit-learn Time (s) |")
    print("|------------|----------|------------------|-------------|------------------------|")
    print(f"| regression | {regression_results['my_mse']:.6f} | {regression_results['sklearn_mse']:.6f} | {regression_results['my_time']:.2f} | {regression_results['sklearn_time']:.2f} |")

if __name__ == "__main__":
    run_all_comparisons() 