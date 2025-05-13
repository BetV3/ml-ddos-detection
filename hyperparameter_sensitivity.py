import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Add the current directory to the path to ensure imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import from our neural network library
from code.network import NeuralNet
from code.layers import Dense
from code.activations import ReLU, Softmax, Tanh, Sigmoid
from code.optimizers import SGD, Adam
from code.losses import cross_entropy_loss
from code.utils import one_hot_encode

def evaluate_learning_rates(X_train, y_train, X_test, y_test, hidden_size=64, batch_size=32, epochs=20):
    """
    Evaluate model performance with different learning rates.
    """
    learning_rates = [0.001, 0.01, 0.05, 0.1]
    results = []
    
    # Get the number of classes from the target data
    num_classes = y_train.shape[1]
    
    for lr in learning_rates:
        print(f"\nTraining with learning rate: {lr}")
        
        # Create model with the current learning rate
        layers = [
            Dense(X_train.shape[1], hidden_size),
            ReLU(),
            Dense(hidden_size, num_classes),  # Ensure output layer matches number of classes
            Softmax()
        ]
        
        optimizer = Adam(learning_rate=lr)
        model = NeuralNet(layers, cross_entropy_loss, optimizer)
        
        # Train model
        history = model.train(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            X_val=X_test,
            y_val=y_test
        )
        
        # Evaluate model
        predictions = model.predict(X_test)
        predicted_classes = np.argmax(predictions, axis=1)
        accuracy = np.mean(predicted_classes == np.argmax(y_test, axis=1)) * 100
        
        print(f"Accuracy with learning rate {lr}: {accuracy:.2f}%")
        
        results.append({
            'learning_rate': lr,
            'accuracy': accuracy,
            'loss_history': history['train_loss']
        })
    
    return results

def evaluate_batch_sizes(X_train, y_train, X_test, y_test, hidden_size=64, learning_rate=0.01, epochs=20):
    """
    Evaluate model performance with different batch sizes.
    """
    batch_sizes = [8, 16, 32, 64, 128]
    results = []
    
    # Get the number of classes from the target data
    num_classes = y_train.shape[1]
    
    for bs in batch_sizes:
        print(f"\nTraining with batch size: {bs}")
        
        # Create model
        layers = [
            Dense(X_train.shape[1], hidden_size),
            ReLU(),
            Dense(hidden_size, num_classes),  # Ensure output layer matches number of classes
            Softmax()
        ]
        
        optimizer = Adam(learning_rate=learning_rate)
        model = NeuralNet(layers, cross_entropy_loss, optimizer)
        
        # Train model
        history = model.train(
            X_train, y_train,
            batch_size=bs,
            epochs=epochs,
            X_val=X_test,
            y_val=y_test
        )
        
        # Evaluate model
        predictions = model.predict(X_test)
        predicted_classes = np.argmax(predictions, axis=1)
        accuracy = np.mean(predicted_classes == np.argmax(y_test, axis=1)) * 100
        
        print(f"Accuracy with batch size {bs}: {accuracy:.2f}%")
        
        results.append({
            'batch_size': bs,
            'accuracy': accuracy,
            'loss_history': history['train_loss']
        })
    
    return results

def evaluate_hidden_sizes(X_train, y_train, X_test, y_test, learning_rate=0.01, batch_size=32, epochs=20):
    """
    Evaluate model performance with different hidden layer sizes.
    """
    hidden_sizes = [16, 32, 64, 128, 256]
    results = []
    
    # Get the number of classes from the target data
    num_classes = y_train.shape[1]
    
    for hs in hidden_sizes:
        print(f"\nTraining with hidden size: {hs}")
        
        # Create model
        layers = [
            Dense(X_train.shape[1], hs),
            ReLU(),
            Dense(hs, num_classes),  # Ensure output layer matches number of classes
            Softmax()
        ]
        
        optimizer = Adam(learning_rate=learning_rate)
        model = NeuralNet(layers, cross_entropy_loss, optimizer)
        
        # Train model
        history = model.train(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            X_val=X_test,
            y_val=y_test
        )
        
        # Evaluate model
        predictions = model.predict(X_test)
        predicted_classes = np.argmax(predictions, axis=1)
        accuracy = np.mean(predicted_classes == np.argmax(y_test, axis=1)) * 100
        
        print(f"Accuracy with hidden size {hs}: {accuracy:.2f}%")
        
        results.append({
            'hidden_size': hs,
            'accuracy': accuracy,
            'loss_history': history['train_loss']
        })
    
    return results

def evaluate_activation_functions(X_train, y_train, X_test, y_test, hidden_size=64, learning_rate=0.01, batch_size=32, epochs=20):
    """
    Evaluate model performance with different activation functions.
    """
    activation_functions = {
        'ReLU': ReLU(),
        'Sigmoid': Sigmoid(),
        'Tanh': Tanh()
    }
    results = []
    
    # Get the number of classes from the target data
    num_classes = y_train.shape[1]
    
    for name, activation in activation_functions.items():
        print(f"\nTraining with activation function: {name}")
        
        # Create model
        layers = [
            Dense(X_train.shape[1], hidden_size),
            activation,
            Dense(hidden_size, num_classes),  # Ensure output layer matches number of classes
            Softmax()
        ]
        
        optimizer = Adam(learning_rate=learning_rate)
        model = NeuralNet(layers, cross_entropy_loss, optimizer)
        
        # Train model
        history = model.train(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            X_val=X_test,
            y_val=y_test
        )
        
        # Evaluate model
        predictions = model.predict(X_test)
        predicted_classes = np.argmax(predictions, axis=1)
        accuracy = np.mean(predicted_classes == np.argmax(y_test, axis=1)) * 100
        
        print(f"Accuracy with {name} activation: {accuracy:.2f}%")
        
        results.append({
            'activation': name,
            'accuracy': accuracy,
            'loss_history': history['train_loss']
        })
    
    return results

def evaluate_optimizers(X_train, y_train, X_test, y_test, hidden_size=64, learning_rate=0.01, batch_size=32, epochs=20):
    """
    Evaluate model performance with different optimizers.
    """
    optimizer_configs = {
        'SGD': SGD(learning_rate=learning_rate),
        'SGD with momentum': SGD(learning_rate=learning_rate, momentum=0.9),
        'Adam': Adam(learning_rate=learning_rate)
    }
    results = []
    
    # Get the number of classes from the target data
    num_classes = y_train.shape[1]
    
    for name, optimizer in optimizer_configs.items():
        print(f"\nTraining with optimizer: {name}")
        
        # Create model
        layers = [
            Dense(X_train.shape[1], hidden_size),
            ReLU(),
            Dense(hidden_size, num_classes),  # Ensure output layer matches number of classes
            Softmax()
        ]
        
        model = NeuralNet(layers, cross_entropy_loss, optimizer)
        
        # Train model
        history = model.train(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            X_val=X_test,
            y_val=y_test
        )
        
        # Evaluate model
        predictions = model.predict(X_test)
        predicted_classes = np.argmax(predictions, axis=1)
        accuracy = np.mean(predicted_classes == np.argmax(y_test, axis=1)) * 100
        
        print(f"Accuracy with {name} optimizer: {accuracy:.2f}%")
        
        results.append({
            'optimizer': name,
            'accuracy': accuracy,
            'loss_history': history['train_loss']
        })
    
    return results

def plot_results(results, param_name, title):
    """
    Plot accuracy and learning curves for different parameter values.
    """
    # Plot accuracy comparison
    plt.figure(figsize=(15, 6))
    
    # Accuracy plot
    plt.subplot(1, 2, 1)
    param_values = [r[param_name] for r in results]
    accuracies = [r['accuracy'] for r in results]
    
    plt.plot(param_values, accuracies, 'o-', linewidth=2)
    plt.xlabel(param_name.replace('_', ' ').title())
    plt.ylabel('Accuracy (%)')
    plt.title(f'Accuracy vs {param_name.replace("_", " ").title()}')
    plt.grid(True)
    
    # Learning curves
    plt.subplot(1, 2, 2)
    for i, result in enumerate(results):
        plt.plot(result['loss_history'], label=f"{param_name}={result[param_name]}")
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Learning Curves')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{param_name}_analysis.png")
    print(f"Results plot saved as '{param_name}_analysis.png'")

if __name__ == "__main__":
    # Load and preprocess data
    print("Loading and preprocessing MNIST digits dataset...")
    digits = load_digits()
    X = digits.data / 16.0  # Normalize to [0,1]
    y = digits.target
    num_classes = len(np.unique(y))
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # One-hot encode targets
    y_train_one_hot = one_hot_encode(y_train, num_classes)
    y_test_one_hot = one_hot_encode(y_test, num_classes)
    
    # Print the shapes to verify
    print(f"X_train shape: {X_train.shape}, y_train_one_hot shape: {y_train_one_hot.shape}")
    print(f"Number of classes: {num_classes}")
    
    # Run hyperparameter sensitivity analysis
    print("\n===== Learning Rate Sensitivity Analysis =====")
    lr_results = evaluate_learning_rates(X_train, y_train_one_hot, X_test, y_test_one_hot)
    plot_results(lr_results, 'learning_rate', 'Learning Rate Sensitivity')
    
    print("\n===== Batch Size Sensitivity Analysis =====")
    bs_results = evaluate_batch_sizes(X_train, y_train_one_hot, X_test, y_test_one_hot)
    plot_results(bs_results, 'batch_size', 'Batch Size Sensitivity')
    
    print("\n===== Hidden Size Sensitivity Analysis =====")
    hs_results = evaluate_hidden_sizes(X_train, y_train_one_hot, X_test, y_test_one_hot)
    plot_results(hs_results, 'hidden_size', 'Hidden Size Sensitivity')
    
    print("\n===== Activation Function Comparison =====")
    act_results = evaluate_activation_functions(X_train, y_train_one_hot, X_test, y_test_one_hot)
    
    # Special plot for categorical values (activation functions)
    plt.figure(figsize=(15, 6))
    
    # Accuracy plot
    plt.subplot(1, 2, 1)
    activations = [r['activation'] for r in act_results]
    accuracies = [r['accuracy'] for r in act_results]
    
    plt.bar(activations, accuracies)
    plt.xlabel('Activation Function')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy by Activation Function')
    
    # Learning curves
    plt.subplot(1, 2, 2)
    for result in act_results:
        plt.plot(result['loss_history'], label=result['activation'])
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Learning Curves by Activation Function')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("activation_analysis.png")
    print("Activation function comparison saved as 'activation_analysis.png'")
    
    print("\n===== Optimizer Comparison =====")
    opt_results = evaluate_optimizers(X_train, y_train_one_hot, X_test, y_test_one_hot)
    
    # Special plot for categorical values (optimizers)
    plt.figure(figsize=(15, 6))
    
    # Accuracy plot
    plt.subplot(1, 2, 1)
    optimizers = [r['optimizer'] for r in opt_results]
    accuracies = [r['accuracy'] for r in opt_results]
    
    plt.bar(optimizers, accuracies)
    plt.xlabel('Optimizer')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy by Optimizer')
    plt.xticks(rotation=15)
    
    # Learning curves
    plt.subplot(1, 2, 2)
    for result in opt_results:
        plt.plot(result['loss_history'], label=result['optimizer'])
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Learning Curves by Optimizer')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("optimizer_analysis.png")
    print("Optimizer comparison saved as 'optimizer_analysis.png'")
    
    print("\nAll hyperparameter sensitivity analyses completed!") 