"""
Script to test the neural network with the full MNIST digits dataset (70,000 28x28 images).
Includes comparisons between 1-layer, 2-layer and scikit-learn's MLPClassifier.
"""
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
import urllib.request
import gzip
import pickle
import time

# Add the current directory to the path to ensure imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import from our neural network library
from code.network import NeuralNet
from code.layers import Dense
from code.activations import ReLU, Softmax
from code.optimizers import Adam
from code.losses import cross_entropy_loss
from code.utils import one_hot_encode

def plot_learning_curve(history, model_name="model"):
    """Plot the learning curve."""
    plt.figure(figsize=(10, 5))
    
    plt.plot(history['train_loss'], label='Training Loss')
    if 'val_loss' in history and history['val_loss']:
        plt.plot(history['val_loss'], label='Validation Loss')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Learning Curve - {model_name}')
    plt.legend()
    plt.grid(True)
    
    plt.savefig(f'mnist_full_{model_name}_learning_curve.png')
    print(f"Learning curve saved as 'mnist_full_{model_name}_learning_curve.png'")

def visualize_predictions(X_test, true_labels, predicted_labels, n_samples=5, model_name="model"):
    """Visualize some predictions."""
    plt.figure(figsize=(15, 3))
    
    indices = np.random.choice(len(X_test), n_samples, replace=False)
    
    for i, idx in enumerate(indices):
        plt.subplot(1, n_samples, i+1)
        plt.imshow(X_test[idx].reshape(28, 28), cmap='gray')
        plt.title(f"True: {true_labels[idx]}\nPred: {predicted_labels[idx]}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(f'mnist_full_{model_name}_prediction_examples.png')
    print(f"Prediction examples saved as 'mnist_full_{model_name}_prediction_examples.png'")

def plot_confusion_matrix(confusion_matrix, model_name="model"):
    """Plot confusion matrix as heatmap."""
    plt.figure(figsize=(10, 8))
    plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.colorbar()
    
    classes = np.arange(10)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    
    # Add text annotations
    thresh = confusion_matrix.max() / 2.
    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            plt.text(j, i, format(confusion_matrix[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if confusion_matrix[i, j] > thresh else "black")
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(f'mnist_full_{model_name}_confusion_matrix.png')
    print(f"Confusion matrix saved as 'mnist_full_{model_name}_confusion_matrix.png'")

def load_mnist():
    """
    Load the full MNIST dataset (70,000 28x28 images).
    Returns:
        X_train, y_train - 60,000 training samples
        X_test, y_test - 10,000 test samples
    """
    # URLs for the MNIST dataset
    urls = [
        'https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz',
        'https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz',
        'https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz',
        'https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz'
    ]
    
    # File paths to save the downloaded files
    file_paths = [
        'mnist_data/train-images-idx3-ubyte.gz',
        'mnist_data/train-labels-idx1-ubyte.gz',
        'mnist_data/t10k-images-idx3-ubyte.gz',
        'mnist_data/t10k-labels-idx1-ubyte.gz'
    ]
    
    # Create the directory if it doesn't exist
    os.makedirs('mnist_data', exist_ok=True)
    
    # Download files if they don't exist
    for url, file_path in zip(urls, file_paths):
        if not os.path.exists(file_path):
            print(f"Downloading {url}...")
            urllib.request.urlretrieve(url, file_path)
    
    # Function to read images
    def extract_images(file_path):
        with gzip.open(file_path, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        return data.reshape(-1, 28*28) / 255.0  # Normalize to [0,1]
    
    # Function to read labels
    def extract_labels(file_path):
        with gzip.open(file_path, 'rb') as f:
            return np.frombuffer(f.read(), np.uint8, offset=8)
    
    # Load the dataset
    X_train = extract_images(file_paths[0])
    y_train = extract_labels(file_paths[1])
    X_test = extract_images(file_paths[2])
    y_test = extract_labels(file_paths[3])
    
    return X_train, y_train, X_test, y_test

def train_single_layer_model(X_train, y_train_one_hot, X_test, y_test_one_hot):
    """Train a model with a single hidden layer."""
    print("\n========== Training Single Hidden Layer Model ==========")
    # Define model architecture
    layers = [
        Dense(X_train.shape[1], 128),  # Input → 128 neurons
        ReLU(),
        Dense(128, 10),               # 128 → 10 output neurons
        Softmax()
    ]
    
    # Create optimizer
    optimizer = Adam(learning_rate=0.001)
    
    # Create neural network
    model = NeuralNet(layers, cross_entropy_loss, optimizer)
    
    # Train model
    print("Training single hidden layer model...")
    start_time = time.time()
    history = model.train(
        X_train, y_train_one_hot,
        batch_size=128,
        epochs=20,
        X_val=X_test, 
        y_val=y_test_one_hot
    )
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Plot learning curve
    plot_learning_curve(history, "single_layer")
    
    return model, history

def train_double_layer_model(X_train, y_train_one_hot, X_test, y_test_one_hot):
    """Train a model with two hidden layers."""
    print("\n========== Training Two Hidden Layer Model ==========")
    # Define model architecture
    layers = [
        Dense(X_train.shape[1], 128),  # Input → 128 neurons
        ReLU(),
        Dense(128, 64),               # 128 → 64 neurons
        ReLU(),
        Dense(64, 10),                # 64 → 10 output neurons
        Softmax()
    ]
    
    # Create optimizer
    optimizer = Adam(learning_rate=0.001)
    
    # Create neural network
    model = NeuralNet(layers, cross_entropy_loss, optimizer)
    
    # Train model
    print("Training two hidden layer model...")
    start_time = time.time()
    history = model.train(
        X_train, y_train_one_hot,
        batch_size=128,
        epochs=20,
        X_val=X_test, 
        y_val=y_test_one_hot
    )
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Plot learning curve
    plot_learning_curve(history, "double_layer")
    
    return model, history

def train_sklearn_model(X_train, y_train, X_test, y_test):
    """Train scikit-learn's MLPClassifier for comparison."""
    print("\n========== Training scikit-learn MLPClassifier ==========")
    
    # Create and train the model
    sklearn_model = MLPClassifier(
        hidden_layer_sizes=(128,),  # Single hidden layer with 128 neurons
        activation='relu',
        solver='adam',
        learning_rate_init=0.001,
        max_iter=20,
        batch_size=128,
        random_state=42
    )
    
    start_time = time.time()
    sklearn_model.fit(X_train, y_train)
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    return sklearn_model

def evaluate_model(model, X_test, y_test, y_test_one_hot=None, is_sklearn=False, model_name="model"):
    """Evaluate model performance."""
    print(f"\n========== Evaluating {model_name} ==========")
    
    # Make predictions
    if is_sklearn:
        y_pred = model.predict(X_test)
    else:
        predictions = model.predict(X_test)
        y_pred = np.argmax(predictions, axis=1)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred) * 100
    print(f"Accuracy on test set: {accuracy:.2f}%")
    
    # Generate confusion matrix
    confusion = np.zeros((10, 10), dtype=int)
    for i in range(len(y_test)):
        confusion[y_test[i], y_pred[i]] += 1
    
    print("\nConfusion Matrix:")
    print(confusion)
    
    # Plot confusion matrix
    plot_confusion_matrix(confusion, model_name)
    
    # Visualize some predictions if not sklearn
    if not is_sklearn:
        visualize_predictions(X_test, y_test, y_pred, model_name=model_name)
    
    return accuracy, confusion

def compare_models(accuracies):
    """Compare model accuracies with a bar chart."""
    plt.figure(figsize=(10, 6))
    models = list(accuracies.keys())
    accs = list(accuracies.values())
    
    plt.bar(models, accs, color=['blue', 'green', 'red'])
    plt.ylabel('Accuracy (%)')
    plt.title('Model Accuracy Comparison on MNIST')
    plt.ylim(95, 100)  # Set y-axis to focus on the relevant range
    
    # Add accuracy values on top of bars
    for i, v in enumerate(accs):
        plt.text(i, v + 0.1, f"{v:.2f}%", ha='center')
    
    plt.tight_layout()
    plt.savefig('mnist_model_comparison.png')
    print("Model comparison chart saved as 'mnist_model_comparison.png'")

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Load full MNIST dataset (60k training, 10k test examples)
    print("Loading full MNIST digits dataset (70,000 28x28 images)...")
    X_train, y_train, X_test, y_test = load_mnist()
    
    # Get dataset information
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    print(f"Number of classes: {len(np.unique(y_train))}")
    
    # Convert labels to one-hot encoded format for our models
    num_classes = len(np.unique(y_train))
    y_train_one_hot = one_hot_encode(y_train, num_classes)
    y_test_one_hot = one_hot_encode(y_test, num_classes)
    
    print(f"X_train shape: {X_train.shape}, y_train_one_hot shape: {y_train_one_hot.shape}")
    
    # 1. Train single hidden layer model
    single_layer_model, single_layer_history = train_single_layer_model(
        X_train, y_train_one_hot, X_test, y_test_one_hot
    )
    
    # 2. Train double hidden layer model
    double_layer_model, double_layer_history = train_double_layer_model(
        X_train, y_train_one_hot, X_test, y_test_one_hot
    )
    
    # 3. Train scikit-learn model
    sklearn_model = train_sklearn_model(X_train, y_train, X_test, y_test)
    
    # Evaluate all models
    single_layer_acc, _ = evaluate_model(
        single_layer_model, X_test, y_test, y_test_one_hot, 
        is_sklearn=False, model_name="single_layer"
    )
    
    double_layer_acc, _ = evaluate_model(
        double_layer_model, X_test, y_test, y_test_one_hot, 
        is_sklearn=False, model_name="double_layer"
    )
    
    sklearn_acc, _ = evaluate_model(
        sklearn_model, X_test, y_test, 
        is_sklearn=True, model_name="sklearn"
    )
    
    # Compare models
    accuracies = {
        "Single Layer": single_layer_acc,
        "Double Layer": double_layer_acc,
        "scikit-learn": sklearn_acc
    }
    
    compare_models(accuracies)
    
    print("\nAll tests completed successfully!") 