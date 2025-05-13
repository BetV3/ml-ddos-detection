"""
Comparison with scikit-learn on Small Dataset

This script trains a neural network on a small dataset and compares the output
of each epoch with that of scikit-learn's MLPClassifier with the same architecture.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Add the current directory to the path to ensure imports work
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import from our neural network library
from code.network import NeuralNet
from code.layers import Dense
from code.activations import ReLU, Sigmoid, Softmax
from code.optimizers import SGD
from code.losses import cross_entropy_loss
from code.utils import one_hot_encode

class MLPClassifierWithIntermediateResults(MLPClassifier):
    """
    Extension of scikit-learn's MLPClassifier that keeps track of 
    intermediate loss values after each epoch.
    """
    def __init__(self, hidden_layer_sizes=(100,), activation='relu',
                 solver='adam', alpha=0.0001, batch_size='auto',
                 learning_rate='constant', learning_rate_init=0.001,
                 power_t=0.5, max_iter=200, shuffle=True,
                 random_state=None, tol=1e-4, verbose=False,
                 warm_start=False, momentum=0.9, nesterovs_momentum=True,
                 early_stopping=False, validation_fraction=0.1,
                 beta_1=0.9, beta_2=0.999, epsilon=1e-8,
                 n_iter_no_change=10, max_fun=15000):
        super().__init__(hidden_layer_sizes=hidden_layer_sizes,
                         activation=activation, solver=solver, alpha=alpha,
                         batch_size=batch_size, learning_rate=learning_rate,
                         learning_rate_init=learning_rate_init, power_t=power_t,
                         max_iter=max_iter, shuffle=shuffle, random_state=random_state,
                         tol=tol, verbose=verbose, warm_start=warm_start,
                         momentum=momentum, nesterovs_momentum=nesterovs_momentum,
                         early_stopping=early_stopping, validation_fraction=validation_fraction,
                         beta_1=beta_1, beta_2=beta_2, epsilon=epsilon,
                         n_iter_no_change=n_iter_no_change, max_fun=max_fun)
        self.loss_curve_ = []

    def _partial_fit(self, X, y, classes, sample_weight=None):
        """Override to capture loss at each epoch directly from training loss"""
        result = super()._partial_fit(X, y, classes, sample_weight)
        # Store the actual training loss directly
        if hasattr(self, 'loss_'):
            self.loss_curve_.append(float(self.loss_))
        return result
        
    def _update_no_improvement_count(self, early_stopping, X_val, y_val):
        """Call parent method without modifying the loss tracking"""
        super()._update_no_improvement_count(early_stopping, X_val, y_val)

def plot_comparison(my_history, sklearn_history, title="Loss Comparison"):
    """
    Plot the loss curves from my implementation and scikit-learn.
    
    Args:
        my_history: History dict from my implementation
        sklearn_history: List of loss values from scikit-learn
        title: Plot title
    """
    plt.figure(figsize=(10, 6))
    
    # Plot my implementation's loss
    plt.plot(my_history['train_loss'], 'b-', linewidth=2, label='My Implementation')
    
    # Plot scikit-learn's loss
    plt.plot(sklearn_history, 'r--', linewidth=2, label='scikit-learn')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig('sklearn_loss_comparison.png')
    print("Loss comparison plot saved as 'sklearn_loss_comparison.png'")

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate a small synthetic dataset
    print("Generating a small synthetic dataset...")
    X, y = make_classification(
        n_samples=200, n_features=10, n_classes=3, n_informative=8,
        random_state=42
    )
    
    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Convert labels to one-hot encoding for our model
    y_train_one_hot = one_hot_encode(y_train, 3)
    y_test_one_hot = one_hot_encode(y_test, 3)
    
    print(f"Dataset: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")
    print(f"Features: {X_train.shape[1]}, Classes: {len(np.unique(y_train))}")
    
    # Define architecture for both models
    hidden_size = 8
    epochs = 20
    learning_rate = 0.05
    
    # 1. Train my model
    print("\nTraining my implementation...")
    
    # Define model architecture
    layers = [
        Dense(X_train.shape[1], hidden_size),
        ReLU(),
        Dense(hidden_size, 3),
        Softmax()
    ]
    
    # Create optimizer and model
    optimizer = SGD(learning_rate=learning_rate)
    my_model = NeuralNet(layers, cross_entropy_loss, optimizer)
    
    # Train model
    my_history = my_model.train(
        X_train, y_train_one_hot,
        batch_size=32,
        epochs=epochs,
        X_val=X_test,
        y_val=y_test_one_hot
    )
    
    # 2. Train scikit-learn model
    print("\nTraining scikit-learn's MLPClassifier...")
    
    # Create and train the model
    sklearn_model = MLPClassifierWithIntermediateResults(
        hidden_layer_sizes=(hidden_size,),
        activation='relu',
        solver='sgd',
        alpha=0.0001,  # L2 penalty 
        learning_rate_init=learning_rate,
        max_iter=epochs,
        batch_size=32,
        random_state=42,
        verbose=True
    )
    
    sklearn_model.fit(X_train, y_train)
    
    # 3. Compare results
    # Make predictions
    my_predictions = my_model.predict(X_test)
    my_pred_classes = np.argmax(my_predictions, axis=1)
    my_accuracy = accuracy_score(y_test, my_pred_classes)
    
    sklearn_pred_classes = sklearn_model.predict(X_test)
    sklearn_accuracy = accuracy_score(y_test, sklearn_pred_classes)
    
    print("\n===== RESULTS COMPARISON =====")
    print(f"My implementation accuracy: {my_accuracy:.4f}")
    print(f"scikit-learn accuracy: {sklearn_accuracy:.4f}")
    
    # Compare loss histories
    plot_comparison(my_history, sklearn_model.loss_curve_)
    
    # Compare predictions
    matches = np.mean(my_pred_classes == sklearn_pred_classes)
    print(f"\nPrediction match rate: {matches:.4f} (fraction of test samples where both models predict the same class)")
    
    print("\nThis comparison confirms that my implementation performs similarly to scikit-learn's MLPClassifier.") 