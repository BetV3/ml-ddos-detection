import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import time
from sklearn.datasets import load_digits, load_iris, load_breast_cancer, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import accuracy_score, mean_squared_error

# Add the current directory to the path to ensure imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import from our neural network library
from code.network import NeuralNet
from code.layers import Dense, Dropout, BatchNorm
from code.activations import ReLU, Sigmoid, Softmax, Tanh
from code.optimizers import SGD, Adam
from code.losses import mse_loss, cross_entropy_loss, binary_cross_entropy_loss
from code.utils import one_hot_encode, normalize

def run_classification_benchmark(dataset_name, X, y, hidden_sizes=[64], dropout_rate=0.0, learning_rate=0.01, 
                                epochs=50, batch_size=32, use_batch_norm=False, activation='relu', runs=3):
    """Run classification benchmark comparing our library against sklearn's MLPClassifier."""
    
    # Determine number of classes
    num_classes = len(np.unique(y))
    is_binary = num_classes == 2
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Normalize data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Prepare target data for our library
    if is_binary:
        # For binary classification with a single output neuron
        Y_train = y_train.reshape(-1, 1)
        Y_test = y_test.reshape(-1, 1)
        loss_fn = binary_cross_entropy_loss
        output_size = 1
        final_activation = Sigmoid()
    else:
        # For multi-class classification with softmax
        Y_train = one_hot_encode(y_train, num_classes)
        Y_test = one_hot_encode(y_test, num_classes)
        loss_fn = cross_entropy_loss
        output_size = num_classes
        final_activation = Softmax()
    
    # Print shape information for debugging
    print(f"X_train shape: {X_train_scaled.shape}, Y_train shape: {Y_train.shape}")
    print(f"Number of classes: {num_classes}, Output size: {output_size}")
    
    # Store results across multiple runs
    scratch_accuracies = []
    sklearn_accuracies = []
    scratch_train_times = []
    sklearn_train_times = []
    
    for run in range(runs):
        print(f"\nRun {run+1}/{runs}:")
        
        # 1. Our implementation
        print(f"Training our MLP on {dataset_name}...")
        
        # Define model architecture
        layers = []
        
        # Input layer
        input_size = X_train.shape[1]
        
        # Build hidden layers
        prev_size = input_size
        for h_size in hidden_sizes:
            layers.append(Dense(prev_size, h_size))
            
            # Optionally add batch normalization
            if use_batch_norm:
                layers.append(BatchNorm(h_size))
                
            # Add activation
            if activation == 'relu':
                layers.append(ReLU())
            elif activation == 'tanh':
                layers.append(Tanh())
                
            # Optionally add dropout
            if dropout_rate > 0:
                layers.append(Dropout(dropout_rate))
                
            prev_size = h_size
        
        # Output layer
        layers.append(Dense(prev_size, output_size))
        layers.append(final_activation)
        
        # Create optimizer and model
        optimizer = Adam(learning_rate=learning_rate)
        scratch_model = NeuralNet(layers, loss_fn, optimizer)
        
        # Train and time
        start_time = time()
        history = scratch_model.train(
            X_train_scaled, Y_train,
            batch_size=batch_size,
            epochs=epochs,
            X_val=X_test_scaled,
            y_val=Y_test
        )
        scratch_train_time = time() - start_time
        scratch_train_times.append(scratch_train_time)
        
        # Make predictions
        predictions = scratch_model.predict(X_test_scaled)
        
        # Calculate accuracy
        if is_binary:
            # For binary classification with sigmoid
            predicted_classes = (predictions > 0.5).astype(int).flatten()
        else:
            # For multi-class with softmax
            predicted_classes = np.argmax(predictions, axis=1)
            
        scratch_accuracy = accuracy_score(y_test, predicted_classes) * 100
        scratch_accuracies.append(scratch_accuracy)
        print(f"Our MLP Accuracy: {scratch_accuracy:.2f}%, Training time: {scratch_train_time:.2f}s")
        
        # 2. scikit-learn MLPClassifier
        print(f"Training sklearn MLPClassifier on {dataset_name}...")
        
        # Set activation function
        sklearn_activation = activation
        
        # Create model and time
        sklearn_mlp = MLPClassifier(
            hidden_layer_sizes=hidden_sizes,
            activation=sklearn_activation,
            solver='adam',
            alpha=0.0001,  # L2 penalty
            batch_size=batch_size,
            learning_rate_init=learning_rate,
            max_iter=epochs,
            random_state=42
        )
        
        start_time = time()
        sklearn_mlp.fit(X_train_scaled, y_train)
        sklearn_train_time = time() - start_time
        sklearn_train_times.append(sklearn_train_time)
        
        # Get accuracy
        sklearn_accuracy = sklearn_mlp.score(X_test_scaled, y_test) * 100
        sklearn_accuracies.append(sklearn_accuracy)
        print(f"sklearn MLPClassifier Accuracy: {sklearn_accuracy:.2f}%, Training time: {sklearn_train_time:.2f}s")
    
    # Average results
    avg_scratch_accuracy = np.mean(scratch_accuracies)
    avg_sklearn_accuracy = np.mean(sklearn_accuracies)
    avg_scratch_time = np.mean(scratch_train_times)
    avg_sklearn_time = np.mean(sklearn_train_times)
    
    # Comparison summary
    results = pd.DataFrame({
        'Implementation': ['Our MLP', 'sklearn MLP'],
        'Avg Accuracy (%)': [round(avg_scratch_accuracy, 2), round(avg_sklearn_accuracy, 2)],
        'Avg Training Time (s)': [round(avg_scratch_time, 2), round(avg_sklearn_time, 2)]
    })
    
    print(f"\nBenchmark Results for {dataset_name} (averaged over {runs} runs):")
    print(results.to_string(index=False))
    
    return {
        'dataset': dataset_name,
        'our_accuracy': avg_scratch_accuracy,
        'sklearn_accuracy': avg_sklearn_accuracy,
        'our_training_time': avg_scratch_time,
        'sklearn_training_time': avg_sklearn_time,
        'our_loss_history': history['train_loss'] if 'train_loss' in history else None
    }

def run_regression_benchmark(dataset_name, X, y, hidden_sizes=[64, 32], learning_rate=0.01, 
                            epochs=50, batch_size=32, use_batch_norm=False, runs=3):
    """Run regression benchmark comparing our library against sklearn's MLPRegressor."""
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Normalize data
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    
    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()
    
    # Print shape information for debugging
    print(f"X_train shape: {X_train_scaled.shape}, y_train shape: {y_train_scaled.reshape(-1, 1).shape}")
    
    # Store results across multiple runs
    scratch_mses = []
    sklearn_mses = []
    scratch_train_times = []
    sklearn_train_times = []
    
    for run in range(runs):
        print(f"\nRun {run+1}/{runs}:")
        
        # 1. Our implementation
        print(f"Training our MLP on {dataset_name}...")
        
        # Define model architecture
        layers = []
        
        # Input layer
        input_size = X_train.shape[1]
        
        # Build hidden layers
        prev_size = input_size
        for h_size in hidden_sizes:
            layers.append(Dense(prev_size, h_size))
            
            # Optionally add batch normalization
            if use_batch_norm:
                layers.append(BatchNorm(h_size))
                
            # Add activation
            layers.append(ReLU())
            
            prev_size = h_size
        
        # Output layer with single neuron for regression
        layers.append(Dense(prev_size, 1))
        
        # Create optimizer and model
        optimizer = Adam(learning_rate=learning_rate)
        scratch_model = NeuralNet(layers, mse_loss, optimizer)
        
        # Train and time
        start_time = time()
        history = scratch_model.train(
            X_train_scaled, 
            y_train_scaled.reshape(-1, 1),
            batch_size=batch_size,
            epochs=epochs
        )
        scratch_train_time = time() - start_time
        scratch_train_times.append(scratch_train_time)
        
        # Make predictions
        predictions = scratch_model.predict(X_test_scaled)
        
        # Inverse transform predictions and calculate MSE
        pred_orig_scale = scaler_y.inverse_transform(predictions).flatten()
        scratch_mse = mean_squared_error(y_test, pred_orig_scale)
        scratch_mses.append(scratch_mse)
        print(f"Our MLP MSE: {scratch_mse:.4f}, Training time: {scratch_train_time:.2f}s")
        
        # 2. scikit-learn MLPRegressor
        print(f"Training sklearn MLPRegressor on {dataset_name}...")
        
        # Create model and time
        sklearn_mlp = MLPRegressor(
            hidden_layer_sizes=hidden_sizes,
            activation='relu',
            solver='adam',
            alpha=0.0001,  # L2 penalty
            batch_size=batch_size,
            learning_rate_init=learning_rate,
            max_iter=epochs,
            random_state=42
        )
        
        start_time = time()
        sklearn_mlp.fit(X_train_scaled, y_train_scaled)
        sklearn_train_time = time() - start_time
        sklearn_train_times.append(sklearn_train_time)
        
        # Get MSE
        sklearn_pred = sklearn_mlp.predict(X_test_scaled)
        sklearn_pred_orig = scaler_y.inverse_transform(sklearn_pred.reshape(-1, 1)).flatten()
        sklearn_mse = mean_squared_error(y_test, sklearn_pred_orig)
        sklearn_mses.append(sklearn_mse)
        print(f"sklearn MLPRegressor MSE: {sklearn_mse:.4f}, Training time: {sklearn_train_time:.2f}s")
    
    # Average results
    avg_scratch_mse = np.mean(scratch_mses)
    avg_sklearn_mse = np.mean(sklearn_mses)
    avg_scratch_time = np.mean(scratch_train_times)
    avg_sklearn_time = np.mean(sklearn_train_times)
    
    # Comparison summary
    results = pd.DataFrame({
        'Implementation': ['Our MLP', 'sklearn MLP'],
        'Avg MSE': [round(avg_scratch_mse, 4), round(avg_sklearn_mse, 4)],
        'Avg Training Time (s)': [round(avg_scratch_time, 2), round(avg_sklearn_time, 2)]
    })
    
    print(f"\nBenchmark Results for {dataset_name} (averaged over {runs} runs):")
    print(results.to_string(index=False))
    
    return {
        'dataset': dataset_name,
        'our_mse': avg_scratch_mse,
        'sklearn_mse': avg_sklearn_mse,
        'our_training_time': avg_scratch_time,
        'sklearn_training_time': avg_sklearn_time,
        'our_loss_history': history['train_loss'] if 'train_loss' in history else None
    }

def plot_comparison(all_results):
    """Plot comparison of accuracy/MSE and training time."""
    
    classification_results = [r for r in all_results if 'our_accuracy' in r]
    regression_results = [r for r in all_results if 'our_mse' in r]
    
    # If we have classification results, plot accuracy comparison
    if classification_results:
        plt.figure(figsize=(12, 5))
        
        # Accuracy comparison
        plt.subplot(1, 2, 1)
        datasets = [r['dataset'] for r in classification_results]
        our_acc = [r['our_accuracy'] for r in classification_results]
        sklearn_acc = [r['sklearn_accuracy'] for r in classification_results]
        
        x = np.arange(len(datasets))
        width = 0.35
        
        plt.bar(x - width/2, our_acc, width, label='Our MLP')
        plt.bar(x + width/2, sklearn_acc, width, label='sklearn MLP')
        
        plt.ylabel('Accuracy (%)')
        plt.title('Classification Accuracy Comparison')
        plt.xticks(x, datasets, rotation=45, ha='right')
        plt.legend()
        
        # Training time comparison
        plt.subplot(1, 2, 2)
        our_time = [r['our_training_time'] for r in classification_results]
        sklearn_time = [r['sklearn_training_time'] for r in classification_results]
        
        plt.bar(x - width/2, our_time, width, label='Our MLP')
        plt.bar(x + width/2, sklearn_time, width, label='sklearn MLP')
        
        plt.ylabel('Training Time (s)')
        plt.title('Training Time Comparison')
        plt.xticks(x, datasets, rotation=45, ha='right')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('classification_comparison.png')
        print("Classification comparison plot saved as 'classification_comparison.png'")
    
    # If we have regression results, plot MSE comparison
    if regression_results:
        plt.figure(figsize=(12, 5))
        
        # MSE comparison
        plt.subplot(1, 2, 1)
        datasets = [r['dataset'] for r in regression_results]
        our_mse = [r['our_mse'] for r in regression_results]
        sklearn_mse = [r['sklearn_mse'] for r in regression_results]
        
        x = np.arange(len(datasets))
        width = 0.35
        
        plt.bar(x - width/2, our_mse, width, label='Our MLP')
        plt.bar(x + width/2, sklearn_mse, width, label='sklearn MLP')
        
        plt.ylabel('Mean Squared Error')
        plt.title('Regression MSE Comparison')
        plt.xticks(x, datasets, rotation=45, ha='right')
        plt.legend()
        
        # Training time comparison
        plt.subplot(1, 2, 2)
        our_time = [r['our_training_time'] for r in regression_results]
        sklearn_time = [r['sklearn_training_time'] for r in regression_results]
        
        plt.bar(x - width/2, our_time, width, label='Our MLP')
        plt.bar(x + width/2, sklearn_time, width, label='sklearn MLP')
        
        plt.ylabel('Training Time (s)')
        plt.title('Training Time Comparison')
        plt.xticks(x, datasets, rotation=45, ha='right')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('regression_comparison.png')
        print("Regression comparison plot saved as 'regression_comparison.png'")

def plot_learning_curves(all_results):
    """Plot learning curves for our MLP implementation."""
    
    # Filter results with loss history
    results_with_history = [r for r in all_results if r['our_loss_history'] is not None]
    
    if not results_with_history:
        return
    
    plt.figure(figsize=(10, 6))
    
    for result in results_with_history:
        plt.plot(result['our_loss_history'], label=result['dataset'])
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Learning Curves')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('learning_curves.png')
    print("Learning curves plot saved as 'learning_curves.png'")

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    all_results = []
    
    # Classification Datasets
    
    # 1. Digits dataset (10 classes)
    print("\n===== Digits Dataset (Multi-class classification) =====")
    digits = load_digits()
    result = run_classification_benchmark(
        'Digits', 
        digits.data / 16.0,  # Normalize to [0,1] 
        digits.target,
        hidden_sizes=[64],
        epochs=30,
        batch_size=32,
        runs=1  # Using fewer runs for quicker testing
    )
    all_results.append(result)
    
    # 2. Deeper network on Digits
    print("\n===== Digits Dataset with Deeper Network =====")
    result = run_classification_benchmark(
        'Digits (Deep)', 
        digits.data / 16.0, 
        digits.target,
        hidden_sizes=[128, 64],  # Two hidden layers
        epochs=30,
        batch_size=32,
        runs=1
    )
    all_results.append(result)
    
    # 3. Iris dataset (3 classes)
    print("\n===== Iris Dataset (Multi-class classification) =====")
    iris = load_iris()
    result = run_classification_benchmark(
        'Iris', 
        iris.data, 
        iris.target,
        hidden_sizes=[32],
        epochs=30,
        batch_size=16,
        runs=1
    )
    all_results.append(result)
    
    # 4. Breast Cancer dataset (Binary classification)
    print("\n===== Breast Cancer Dataset (Binary classification) =====")
    cancer = load_breast_cancer()
    result = run_classification_benchmark(
        'Breast Cancer', 
        cancer.data, 
        cancer.target,
        hidden_sizes=[64, 32],
        epochs=30,
        batch_size=32,
        runs=1
    )
    all_results.append(result)
    
    # 5. With regularization (Dropout)
    print("\n===== Digits with Dropout Regularization =====")
    result = run_classification_benchmark(
        'Digits (Dropout)', 
        digits.data / 16.0, 
        digits.target,
        hidden_sizes=[128, 64],
        dropout_rate=0.3,  # Add dropout
        epochs=30,
        batch_size=32,
        runs=1
    )
    all_results.append(result)
    
    # 6. With Batch Normalization
    print("\n===== Digits with Batch Normalization =====")
    result = run_classification_benchmark(
        'Digits (BatchNorm)', 
        digits.data / 16.0, 
        digits.target,
        hidden_sizes=[128, 64],
        use_batch_norm=True,
        epochs=30,
        batch_size=32,
        runs=1
    )
    all_results.append(result)
    
    # Regression Dataset
    
    # 7. California Housing dataset (Regression)
    print("\n===== California Housing Dataset (Regression) =====")
    housing = fetch_california_housing()
    result = run_regression_benchmark(
        'California Housing',
        housing.data,
        housing.target,
        hidden_sizes=[64, 32],
        epochs=50,
        batch_size=64,
        runs=1
    )
    all_results.append(result)
    
    # Generate comparison plots
    plot_comparison(all_results)
    plot_learning_curves(all_results)
    
    print("\nAll benchmarks completed!") 