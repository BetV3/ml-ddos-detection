"""
Simple script to test the neural network with the MNIST digits dataset.
"""
import sys
import os
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Add the current directory to the path to ensure imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import from our neural network library
from code.network import NeuralNet
from code.layers import Dense
from code.activations import ReLU, Softmax
from code.optimizers import Adam
from code.losses import cross_entropy_loss
from code.utils import one_hot_encode

def plot_learning_curve(history):
    """Plot the learning curve."""
    plt.figure(figsize=(10, 5))
    
    plt.plot(history['train_loss'], label='Training Loss')
    if 'val_loss' in history and history['val_loss']:
        plt.plot(history['val_loss'], label='Validation Loss')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Learning Curve')
    plt.legend()
    plt.grid(True)
    
    plt.savefig('simple_test_learning_curve.png')
    print("Learning curve saved as 'simple_test_learning_curve.png'")

def visualize_predictions(X_test, true_labels, predicted_labels, n_samples=5):
    """Visualize some predictions."""
    plt.figure(figsize=(15, 3))
    
    indices = np.random.choice(len(X_test), n_samples, replace=False)
    
    for i, idx in enumerate(indices):
        plt.subplot(1, n_samples, i+1)
        plt.imshow(X_test[idx].reshape(8, 8), cmap='gray')
        plt.title(f"True: {true_labels[idx]}\nPred: {predicted_labels[idx]}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('prediction_examples.png')
    print("Prediction examples saved as 'prediction_examples.png'")

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Load MNIST digits dataset
    print("Loading MNIST digits dataset...")
    digits = load_digits()
    X = digits.data / 16.0  # Normalize to [0,1]
    y = digits.target
    
    # Get dataset information
    print(f"Dataset shape: {X.shape}")
    print(f"Number of classes: {len(np.unique(y))}")
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Convert labels to one-hot encoded format
    num_classes = len(np.unique(y))
    y_train_one_hot = one_hot_encode(y_train, num_classes)
    y_test_one_hot = one_hot_encode(y_test, num_classes)
    
    print(f"X_train shape: {X_train.shape}, y_train_one_hot shape: {y_train_one_hot.shape}")
    
    # Define model architecture
    layers = [
        Dense(X_train.shape[1], 64),
        ReLU(),
        Dense(64, num_classes),
        Softmax()
    ]
    
    # Create optimizer
    optimizer = Adam(learning_rate=0.01)
    
    # Create neural network
    model = NeuralNet(layers, cross_entropy_loss, optimizer)
    
    # Train model
    print("Training the model...")
    history = model.train(
        X_train, y_train_one_hot,
        batch_size=32,
        epochs=20,
        X_val=X_test, 
        y_val=y_test_one_hot
    )
    
    # Plot learning curve
    plot_learning_curve(history)
    
    # Make predictions
    print("Making predictions...")
    predictions = model.predict(X_test)
    
    # Check prediction shape
    print(f"Predictions shape: {predictions.shape}")
    
    # Convert to class labels
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = y_test
    
    # Calculate accuracy
    accuracy = accuracy_score(true_classes, predicted_classes) * 100
    print(f"Accuracy: {accuracy:.2f}%")
    
    # Visualize some predictions
    visualize_predictions(X_test, true_classes, predicted_classes)
    
    # Print confusion matrix
    confusion = np.zeros((num_classes, num_classes), dtype=int)
    for i in range(len(true_classes)):
        confusion[true_classes[i], predicted_classes[i]] += 1
    
    print("\nConfusion Matrix:")
    print(confusion)
    
    print("\nSimple test completed successfully!") 