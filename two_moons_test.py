"""
Two Moons Synthetic Dataset Test with Decision Boundary Visualization

This script tests the neural network implementation on the 'two moons' 
synthetic dataset, which consists of two interleaving half circles.
It demonstrates the ability of a neural network to learn non-linear
decision boundaries.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap

# Add the current directory to the path to ensure imports work
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import from our neural network library
from code.network import NeuralNet
from code.layers import Dense
from code.activations import ReLU, Sigmoid
from code.optimizers import Adam, SGD
from code.losses import binary_cross_entropy_loss
from code.utils import one_hot_encode

def plot_decision_boundary(model, X, y, title="Decision Boundary"):
    """
    Plot the decision boundary of a model.
    
    Args:
        model: Trained model with predict method
        X: Input features
        y: Target labels
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
    Z = model.predict(mesh_points)
    
    # For binary classification, we need class predictions (0 or 1)
    if Z.shape[1] > 1:  # One-hot encoded output
        Z = np.argmax(Z, axis=1)
    else:  # Single value output
        Z = (Z > 0.5).astype(int).ravel()
    
    # Reshape the predictions
    Z = Z.reshape(xx.shape)
    
    # Create plot
    plt.figure(figsize=(10, 8))
    
    # Define custom colormap
    cmap_bg = ListedColormap(['#FFAAAA', '#AAAAFF'])
    cmap_points = ListedColormap(['#FF0000', '#0000FF'])
    
    # Plot decision boundary
    plt.contourf(xx, yy, Z, cmap=cmap_bg, alpha=0.8)
    
    # Plot class samples
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_points, 
                edgecolor='k', s=40)
    
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.tight_layout()
    plt.savefig('two_moons_decision_boundary.png')
    print("Decision boundary plot saved as 'two_moons_decision_boundary.png'")

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate Two Moons dataset
    print("Generating Two Moons synthetic dataset...")
    X, y = make_moons(n_samples=500, noise=0.2, random_state=42)
    
    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Convert labels to one-hot encoding for training
    y_train_one_hot = one_hot_encode(y_train, 2)
    y_test_one_hot = one_hot_encode(y_test, 2)
    
    # Define model architecture
    print("\nBuilding neural network model...")
    layers = [
        Dense(2, 10),  # Input layer (2 features) to hidden layer (10 neurons)
        ReLU(),        # ReLU activation
        Dense(10, 2),  # Output layer (2 classes)
        Sigmoid()      # Sigmoid activation for binary classification
    ]
    
    # Create optimizer and model
    optimizer = Adam(learning_rate=0.01)
    model = NeuralNet(layers, binary_cross_entropy_loss, optimizer)
    
    # Train model
    print("\nTraining neural network on Two Moons dataset...")
    history = model.train(
        X_train, y_train_one_hot,
        batch_size=32,
        epochs=1000,  # Increased to 1000 iterations as mentioned in your description
        X_val=X_test,
        y_val=y_test_one_hot
    )
    
    # Evaluate model
    train_predictions = model.predict(X_train)
    train_pred_classes = np.argmax(train_predictions, axis=1)
    train_accuracy = np.mean(train_pred_classes == y_train)
    
    test_predictions = model.predict(X_test)
    test_pred_classes = np.argmax(test_predictions, axis=1)
    test_accuracy = np.mean(test_pred_classes == y_test)
    
    print(f"\nTraining Accuracy: {train_accuracy:.4f}")
    print(f"Testing Accuracy: {test_accuracy:.4f}")
    
    # Plot decision boundary
    plot_decision_boundary(model, X, y, "Two Moons Classification with Neural Network")
    
    # Plot learning curve
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Training Loss')
    if 'val_loss' in history:
        plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Learning Curve for Two Moons Classification')
    plt.legend()
    plt.grid(True)
    plt.savefig('two_moons_learning_curve.png')
    print("Learning curve saved as 'two_moons_learning_curve.png'")
    
    print("\nTwo Moons classification test completed!") 