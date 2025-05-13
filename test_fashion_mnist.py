"""
Script to test neural network implementations on the Fashion-MNIST dataset.
Includes:
1. MLP with 128 hidden neurons
2. Simple CNN with one convolutional layer (6 filters)
3. More complex CNN with two convolutional layers (32 and 64 filters)
4. Visualization of learned convolutional filters
"""
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
import urllib.request
import gzip
import time

# Add the current directory to the path to ensure imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import from our neural network library
from code.network import NeuralNet
from code.layers import Dense, Conv2D, MaxPool, Flatten
from code.activations import ReLU, Softmax
from code.optimizers import SGD, Adam
from code.losses import cross_entropy_loss
from code.utils import one_hot_encode

# Fashion-MNIST class names for reference
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

def plot_learning_curve(history, model_name="model"):
    """Plot the learning curve."""
    plt.figure(figsize=(12, 5))
    
    # Training and validation loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    if 'val_loss' in history and history['val_loss']:
        plt.plot(history['val_loss'], label='Validation Loss')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Learning Curve - {model_name}')
    plt.legend()
    plt.grid(True)
    
    # Accuracy if available
    if 'train_acc' in history and history['train_acc']:
        plt.subplot(1, 2, 2)
        plt.plot(history['train_acc'], label='Training Accuracy')
        if 'val_acc' in history and history['val_acc']:
            plt.plot(history['val_acc'], label='Validation Accuracy')
        
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title(f'Accuracy Curve - {model_name}')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'fashion_mnist_{model_name}_learning_curve.png')
    print(f"Learning curve saved as 'fashion_mnist_{model_name}_learning_curve.png'")

def visualize_predictions(X_test, true_labels, predicted_labels, n_samples=10, model_name="model"):
    """Visualize some predictions."""
    plt.figure(figsize=(15, 8))
    
    # Get n_samples random indices, but ensure we have at least one of each class if possible
    indices = []
    for class_idx in range(10):
        class_indices = np.where(true_labels == class_idx)[0]
        if len(class_indices) > 0:
            indices.append(np.random.choice(class_indices))
    
    # Fill the rest with random samples
    remaining = n_samples - len(indices)
    if remaining > 0:
        random_indices = np.random.choice(
            np.arange(len(X_test)), 
            size=remaining, 
            replace=False
        )
        indices.extend(random_indices)
    
    # Ensure we don't exceed n_samples
    indices = indices[:n_samples]
    
    # Plot each sample
    for i, idx in enumerate(indices):
        plt.subplot(2, n_samples//2, i+1)
        plt.imshow(X_test[idx].reshape(28, 28), cmap='gray')
        plt.title(f"True: {class_names[true_labels[idx]]}\nPred: {class_names[predicted_labels[idx]]}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(f'fashion_mnist_{model_name}_predictions.png')
    print(f"Predictions visualization saved as 'fashion_mnist_{model_name}_predictions.png'")

def plot_confusion_matrix(cm, model_name="model"):
    """Plot confusion matrix."""
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.colorbar()
    
    # Add axis labels
    tick_marks = np.arange(10)
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(f'fashion_mnist_{model_name}_confusion_matrix.png')
    print(f"Confusion matrix saved as 'fashion_mnist_{model_name}_confusion_matrix.png'")

def visualize_filters(model, model_name="cnn"):
    """Visualize the learned convolutional filters from the first layer."""
    # Get the first convolutional layer
    conv_layer = None
    for layer in model.layers:
        if isinstance(layer, Conv2D):
            conv_layer = layer
            break
    
    if conv_layer is None:
        print("No convolutional layer found in the model.")
        return
    
    # Get weights from the layer
    filters = conv_layer.W
    n_filters = filters.shape[0]  # Number of filters
    
    # Plot filters
    plt.figure(figsize=(12, 2))
    for i in range(n_filters):
        plt.subplot(1, n_filters, i+1)
        
        # For each filter channel, take the mean to get a 2D representation
        filter_img = np.mean(filters[i], axis=0)
        
        # Normalize for better visualization
        filter_img = (filter_img - filter_img.min()) / (filter_img.max() - filter_img.min() + 1e-8)
        
        plt.imshow(filter_img, cmap='viridis')
        plt.title(f'Filter {i+1}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(f'fashion_mnist_{model_name}_filters.png')
    print(f"Convolutional filters visualization saved as 'fashion_mnist_{model_name}_filters.png'")

def load_fashion_mnist():
    """
    Load the Fashion-MNIST dataset.
    Returns:
        X_train, y_train - 60,000 training samples
        X_test, y_test - 10,000 test samples
    """
    # URLs for the Fashion-MNIST dataset
    urls = [
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz',
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz',
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz',
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz'
    ]
    
    # File paths to save the downloaded files
    file_paths = [
        'fashion_mnist_data/train-images-idx3-ubyte.gz',
        'fashion_mnist_data/train-labels-idx1-ubyte.gz',
        'fashion_mnist_data/t10k-images-idx3-ubyte.gz',
        'fashion_mnist_data/t10k-labels-idx1-ubyte.gz'
    ]
    
    # Create the directory if it doesn't exist
    os.makedirs('fashion_mnist_data', exist_ok=True)
    
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

def train_mlp(X_train, y_train_one_hot, X_test, y_test_one_hot, num_epochs=20):
    """Train a simple MLP model with 128 hidden neurons."""
    print("\n========== Training MLP Model ==========")
    
    # Define model architecture
    layers = [
        Dense(X_train.shape[1], 128),  # Input → 128 neurons
        ReLU(),
        Dense(128, 10),              # 128 → 10 output neurons
        Softmax()
    ]
    
    # Create optimizer
    optimizer = Adam(learning_rate=0.001)
    
    # Create neural network
    model = NeuralNet(layers, cross_entropy_loss, optimizer)
    
    # Train model
    print("Training MLP model...")
    start_time = time.time()
    history = model.train(
        X_train, y_train_one_hot,
        batch_size=64,
        epochs=num_epochs,
        X_val=X_test, 
        y_val=y_test_one_hot
    )
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Plot learning curve
    plot_learning_curve(history, "mlp")
    
    return model, history

def train_simple_cnn(X_train_cnn, y_train_one_hot, X_test_cnn, y_test_one_hot, num_epochs=200):
    """Train a simple CNN with one conv layer (6 filters) and max pooling."""
    print("\n========== Training Simple CNN Model ==========")
    
    # Reshape data to channels-first format (batch_size, channels, height, width)
    X_train_reshaped = X_train_cnn.transpose(0, 3, 1, 2)
    X_test_reshaped = X_test_cnn.transpose(0, 3, 1, 2)
    
    # Define model architecture - using correct Conv2D parameters
    layers = [
        Conv2D(input_channels=1, num_filters=6, kernel_size=5, stride=1, padding=2),
        ReLU(),
        MaxPool(pool_size=2, stride=2),
        Flatten(),
        Dense(6*14*14, 100),  # Flattened output size after pooling
        ReLU(),
        Dense(100, 10),
        Softmax()
    ]
    
    # Create optimizer (SGD as mentioned in report)
    optimizer = SGD(learning_rate=0.01)
    
    # Create neural network
    model = NeuralNet(layers, cross_entropy_loss, optimizer)
    
    # Train model
    print("Training simple CNN model...")
    start_time = time.time()
    history = model.train(
        X_train_reshaped, y_train_one_hot,
        batch_size=64,
        epochs=num_epochs,
        X_val=X_test_reshaped, 
        y_val=y_test_one_hot
    )
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Plot learning curve
    plot_learning_curve(history, "simple_cnn")
    
    # Visualize filters
    visualize_filters(model, "simple_cnn")
    
    return model, history

def train_complex_cnn(X_train_cnn, y_train_one_hot, X_test_cnn, y_test_one_hot, num_epochs=15):
    """Train a more complex CNN with two conv layers (32 and 64 filters) and max pooling."""
    print("\n========== Training Complex CNN Model ==========")
    
    # Reshape data to channels-first format (batch_size, channels, height, width)
    X_train_reshaped = X_train_cnn.transpose(0, 3, 1, 2)
    X_test_reshaped = X_test_cnn.transpose(0, 3, 1, 2)
    
    # Define model architecture - using correct Conv2D parameters
    layers = [
        Conv2D(input_channels=1, num_filters=32, kernel_size=3, stride=1, padding=1),
        ReLU(),
        MaxPool(pool_size=2, stride=2),
        Conv2D(input_channels=32, num_filters=64, kernel_size=3, stride=1, padding=1),
        ReLU(),
        MaxPool(pool_size=2, stride=2),
        Flatten(),
        Dense(64*7*7, 128),  # Flattened output size after second pooling
        ReLU(),
        Dense(128, 10),
        Softmax()
    ]
    
    # Create optimizer (Adam as mentioned in report)
    optimizer = Adam(learning_rate=0.001)
    
    # Create neural network
    model = NeuralNet(layers, cross_entropy_loss, optimizer)
    
    # Train model
    print("Training complex CNN model...")
    start_time = time.time()
    history = model.train(
        X_train_reshaped, y_train_one_hot,
        batch_size=64,
        epochs=num_epochs,
        X_val=X_test_reshaped, 
        y_val=y_test_one_hot
    )
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Plot learning curve
    plot_learning_curve(history, "complex_cnn")
    
    # Visualize filters
    visualize_filters(model, "complex_cnn")
    
    return model, history

def evaluate_model(model, X_test, y_test, X_test_cnn=None, model_name="model", is_cnn=False):
    """Evaluate model and display results."""
    print(f"\n========== Evaluating {model_name} ==========")
    
    # Use the appropriate input format based on model type
    if is_cnn and X_test_cnn is not None:
        # Reshape to channels-first for CNN models
        X_test_reshaped = X_test_cnn.transpose(0, 3, 1, 2)
        predictions = model.predict(X_test_reshaped)
    else:
        predictions = model.predict(X_test)
    
    # Convert to class labels
    y_pred = np.argmax(predictions, axis=1)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred) * 100
    print(f"Accuracy on test set: {accuracy:.2f}%")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    
    # Visualize results
    plot_confusion_matrix(cm, model_name)
    
    visualize_predictions(X_test, y_test, y_pred, model_name=model_name)
    
    return accuracy, cm

def compare_models(accuracies):
    """Compare model accuracies with a bar chart."""
    plt.figure(figsize=(10, 6))
    models = list(accuracies.keys())
    accs = list(accuracies.values())
    
    plt.bar(models, accs, color=['blue', 'green', 'red'])
    plt.ylabel('Accuracy (%)')
    plt.title('Model Accuracy Comparison on Fashion-MNIST')
    plt.ylim(75, 95)  # Set y-axis to focus on the relevant range
    
    # Add accuracy values on top of bars
    for i, v in enumerate(accs):
        plt.text(i, v + 0.3, f"{v:.2f}%", ha='center')
    
    plt.tight_layout()
    plt.savefig('fashion_mnist_model_comparison.png')
    print("Model comparison chart saved as 'fashion_mnist_model_comparison.png'")

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Load Fashion-MNIST dataset
    print("Loading Fashion-MNIST dataset...")
    X_train, y_train, X_test, y_test = load_fashion_mnist()
    
    # Print dataset information
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    print(f"Number of classes: {len(np.unique(y_train))}")
    
    # Prepare data for CNN format (add channel dimension)
    X_train_cnn = X_train.reshape(-1, 28, 28, 1)
    X_test_cnn = X_test.reshape(-1, 28, 28, 1)
    
    # Convert labels to one-hot encoded format
    num_classes = len(np.unique(y_train))
    y_train_one_hot = one_hot_encode(y_train, num_classes)
    y_test_one_hot = one_hot_encode(y_test, num_classes)
    
    print(f"X_train shape: {X_train.shape}, y_train_one_hot shape: {y_train_one_hot.shape}")
    print(f"X_train_cnn shape: {X_train_cnn.shape}")
    
    # Train MLP model
    mlp_model, mlp_history = train_mlp(
        X_train, y_train_one_hot, X_test, y_test_one_hot, num_epochs=20
    )
    
    # Train simple CNN model - using fewer epochs for testing
    simple_cnn_model, simple_cnn_history = train_simple_cnn(
        X_train_cnn, y_train_one_hot, X_test_cnn, y_test_one_hot, num_epochs=10  # Reduced for testing
    )
    
    # Train complex CNN model
    complex_cnn_model, complex_cnn_history = train_complex_cnn(
        X_train_cnn, y_train_one_hot, X_test_cnn, y_test_one_hot, num_epochs=10  # Reduced for testing
    )
    
    # Evaluate all models
    mlp_acc, _ = evaluate_model(
        mlp_model, X_test, y_test, model_name="mlp"
    )
    
    simple_cnn_acc, _ = evaluate_model(
        simple_cnn_model, X_test, y_test, X_test_cnn, 
        model_name="simple_cnn", is_cnn=True
    )
    
    complex_cnn_acc, _ = evaluate_model(
        complex_cnn_model, X_test, y_test, X_test_cnn, 
        model_name="complex_cnn", is_cnn=True
    )
    
    # Compare models
    accuracies = {
        "MLP (128)": mlp_acc,
        "CNN (1 layer)": simple_cnn_acc,
        "CNN (2 layers)": complex_cnn_acc
    }
    
    compare_models(accuracies)
    
    print("\nAll Fashion-MNIST tests completed successfully!") 