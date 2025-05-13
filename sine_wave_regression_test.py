"""
Sine Wave Regression Test

This script tests the neural network implementation on a noisy sine wave dataset.
It demonstrates the ability of a neural network to approximate a continuous function
with the MSE loss. The model uses two hidden layers with 20 neurons each and ReLU activation.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Add the current directory to the path to ensure imports work
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import from our neural network library
from code.network import NeuralNet
from code.layers import Dense
from code.activations import ReLU
from code.optimizers import Adam
from code.losses import mse_loss

def generate_sine_data(n_samples=500, noise_level=0.2, random_state=42):
    """
    Generate a noisy sine wave dataset in the range [0, 2π].
    
    Args:
        n_samples: Number of data points to generate
        noise_level: Standard deviation of the Gaussian noise
        random_state: Random seed for reproducibility
        
    Returns:
        X: Input features (x values)
        y: Target values (noisy sine values)
    """
    np.random.seed(random_state)
    
    # Generate x values in [0, 2π]
    X = np.linspace(0, 2*np.pi, n_samples).reshape(-1, 1)
    
    # Generate noisy sine values: y = sin(x) + noise
    y = np.sin(X) + noise_level * np.random.randn(n_samples, 1)
    
    return X, y

def plot_regression_results(X, y, y_pred, title="Sine Wave Regression"):
    """
    Plot the ground truth data points and the model's predictions.
    
    Args:
        X: Input features
        y: True target values
        y_pred: Model predictions
        title: Plot title
    """
    plt.figure(figsize=(10, 6))
    
    # Plot ground truth and predictions
    plt.scatter(X, y, alpha=0.7, label='Ground truth (noisy)', color='blue', s=20)
    plt.plot(X, y_pred, 'r-', linewidth=2, label='Model predictions')
    
    # Add a reference pure sine curve
    X_pure = np.linspace(0, 2*np.pi, 1000).reshape(-1, 1)
    y_pure = np.sin(X_pure)
    plt.plot(X_pure, y_pure, 'g--', linewidth=1.5, label='sin(x) (no noise)')
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig('sine_wave_regression.png')
    print("Regression results saved as 'sine_wave_regression.png'")

if __name__ == "__main__":
    # Generate dataset
    print("Generating noisy sine wave dataset...")
    X, y = generate_sine_data(n_samples=500, noise_level=0.2)
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Define model architecture with two hidden layers of 20 neurons each
    print("\nBuilding neural network model...")
    layers = [
        Dense(1, 20),     # Input layer (1 feature) to first hidden layer (20 neurons)
        ReLU(),           # ReLU activation
        Dense(20, 20),    # Second hidden layer (20 neurons)
        ReLU(),           # ReLU activation
        Dense(20, 1)      # Output layer (1 neuron for regression)
    ]
    
    # Create optimizer and model
    optimizer = Adam(learning_rate=0.01)
    model = NeuralNet(layers, mse_loss, optimizer)
    
    # Train model
    print("\nTraining neural network on sine wave dataset...")
    history = model.train(
        X_train, y_train,
        batch_size=32,
        epochs=500,  # Train for a few hundred epochs as mentioned
        X_val=X_test,
        y_val=y_test
    )
    
    # Make predictions
    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)
    
    # Calculate MSE
    train_mse = mean_squared_error(y_train, train_predictions)
    test_mse = mean_squared_error(y_test, test_predictions)
    
    print(f"\nTraining MSE: {train_mse:.6f}")
    print(f"Test MSE: {test_mse:.6f}")
    
    # Sort data for cleaner plotting
    idx = np.argsort(X.flatten())
    X_sorted = X[idx]
    y_sorted = y[idx]
    
    # Make predictions on the full sorted dataset
    y_pred_sorted = model.predict(X_sorted)
    
    # Plot results
    plot_regression_results(X_sorted, y_sorted, y_pred_sorted, 
                          title="Neural Network Regression on Sine Wave")
    
    # Plot learning curve
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Training Loss')
    if 'val_loss' in history:
        plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Learning Curve for Sine Wave Regression')
    plt.legend()
    plt.grid(True)
    plt.savefig('sine_wave_learning_curve.png')
    print("Learning curve saved as 'sine_wave_learning_curve.png'")
    
    print("\nSine wave regression test completed!") 