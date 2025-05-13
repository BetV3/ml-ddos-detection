import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.neural_network import MLPRegressor

# Add the current directory to the path to ensure imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import from our neural network library
from code.network import NeuralNet
from code.layers import Dense
from code.activations import ReLU, Tanh
from code.optimizers import Adam
from code.losses import mse_loss

def create_time_series_data(n_samples=1000):
    """Create a synthetic time series dataset with seasonality and trend."""
    t = np.linspace(0, 4*np.pi, n_samples)
    # Trend component
    trend = 0.1 * t
    # Seasonal component
    seasonal = 2.5 * np.sin(t) + 0.5 * np.sin(3*t)
    # Random noise
    noise = 0.5 * np.random.randn(n_samples)
    # Combine components
    signal = trend + seasonal + noise
    return signal

def create_sequences(data, seq_length):
    """Create input sequences and targets for time series prediction."""
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i+seq_length]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys).reshape(-1, 1)

def plot_predictions(y_true, y_pred_scratch, y_pred_sklearn, sequence_length, title):
    """Plot true values and predictions from both models."""
    plt.figure(figsize=(15, 6))
    
    # Plot only a subset of the data for better visualization
    subset_size = min(500, len(y_true))
    start_idx = sequence_length
    end_idx = start_idx + subset_size
    
    plt.plot(range(start_idx, end_idx), y_true[start_idx:end_idx], 'b-', label='True Values')
    plt.plot(range(start_idx, end_idx), y_pred_scratch[start_idx:end_idx], 'r-', label='Our MLP')
    plt.plot(range(start_idx, end_idx), y_pred_sklearn[start_idx:end_idx], 'g-', label='sklearn MLP')
    
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    
    plt.savefig('time_series_prediction.png')
    print("Time series predictions plot saved as 'time_series_prediction.png'")

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create synthetic time series data
    print("Generating synthetic time series data...")
    data = create_time_series_data(n_samples=1500)
    
    # Normalize data
    scaler = MinMaxScaler(feature_range=(-1, 1))
    data_scaled = scaler.fit_transform(data.reshape(-1, 1)).flatten()
    
    # Define sequence length for time series prediction
    sequence_length = 10
    
    # Create sequences
    X, y = create_sequences(data_scaled, sequence_length)
    
    # Split into train and test sets (80% train, 20% test)
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
    
    # 1. Our MLP implementation
    print("\nTraining our MLP for time series prediction...")
    
    # Define model architecture
    layers = [
        Dense(sequence_length, 32),
        Tanh(),  # Tanh often works well for time series
        Dense(32, 16),
        ReLU(),
        Dense(16, 1)  # Output a single value
    ]
    
    # Create optimizer and model
    optimizer = Adam(learning_rate=0.01)
    our_model = NeuralNet(layers, mse_loss, optimizer)
    
    # Train model
    history = our_model.train(
        X_train, y_train,
        batch_size=32,
        epochs=50
    )
    
    # Make predictions
    y_pred_our = our_model.predict(X_test)
    
    # Check shapes to ensure compatibility
    print(f"Predictions shape: {y_pred_our.shape}, Test targets shape: {y_test.shape}")
    
    # Calculate metrics
    mse_our = mean_squared_error(y_test, y_pred_our)
    mae_our = mean_absolute_error(y_test, y_pred_our)
    
    print(f"Our MLP - MSE: {mse_our:.6f}, MAE: {mae_our:.6f}")
    
    # 2. scikit-learn MLPRegressor
    print("\nTraining sklearn MLPRegressor for time series prediction...")
    
    sklearn_model = MLPRegressor(
        hidden_layer_sizes=(32, 16),
        activation='tanh',  # Using tanh for the first layer
        solver='adam',
        learning_rate_init=0.01,
        max_iter=50,
        random_state=42
    )
    
    sklearn_model.fit(X_train, y_train.ravel())
    
    # Make predictions
    y_pred_sklearn = sklearn_model.predict(X_test).reshape(-1, 1)
    
    # Calculate metrics
    mse_sklearn = mean_squared_error(y_test, y_pred_sklearn)
    mae_sklearn = mean_absolute_error(y_test, y_pred_sklearn)
    
    print(f"sklearn MLPRegressor - MSE: {mse_sklearn:.6f}, MAE: {mae_sklearn:.6f}")
    
    # Comparison summary
    results = pd.DataFrame({
        'Implementation': ['Our MLP', 'sklearn MLP'],
        'MSE': [mse_our, mse_sklearn],
        'MAE': [mae_our, mae_sklearn]
    })
    
    print("\nComparison Results:")
    print(results.to_string(index=False))
    
    # Inverse transform for plotting
    # Create arrays with placeholders for the sequences
    full_data = data_scaled.copy()
    y_pred_full_our = np.zeros_like(full_data)
    y_pred_full_sklearn = np.zeros_like(full_data)
    
    # Fill the prediction arrays with predicted values (only for test portion)
    test_start = train_size + sequence_length
    
    # Check array lengths to avoid index errors
    pred_len = min(len(y_pred_our), len(y_pred_full_our) - test_start)
    y_pred_full_our[test_start:test_start+pred_len] = y_pred_our[:pred_len].flatten()
    y_pred_full_sklearn[test_start:test_start+pred_len] = y_pred_sklearn[:pred_len].flatten()
    
    # Inverse transform
    original_data = scaler.inverse_transform(full_data.reshape(-1, 1)).flatten()
    our_preds_orig = scaler.inverse_transform(y_pred_full_our.reshape(-1, 1)).flatten()
    sklearn_preds_orig = scaler.inverse_transform(y_pred_full_sklearn.reshape(-1, 1)).flatten()
    
    # Plot results
    plot_predictions(
        original_data, 
        our_preds_orig, 
        sklearn_preds_orig, 
        sequence_length,
        'Time Series Prediction Comparison'
    )
    
    # Plot learning curve
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Training Loss Curve')
    plt.grid(True)
    plt.savefig('time_series_learning_curve.png')
    print("Learning curve saved as 'time_series_learning_curve.png'")
    
    print("\nTime series prediction test completed!") 