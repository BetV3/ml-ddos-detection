"""
Numerical Gradient Check for Tiny Network

This script implements a gradient check for a tiny neural network (2 inputs, 2 hidden neurons, 2 outputs).
It compares the backpropagation gradients with numerically computed gradients to verify correctness.
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import from our neural network library
from code.network import NeuralNet
from code.layers import Dense
from code.activations import ReLU, Sigmoid, Softmax
from code.losses import cross_entropy_loss, mse_loss
from code.optimizers import SGD

def compute_numerical_gradient(model, X, y, eps=1e-7):
    """
    Compute numerical gradients for all parameters of the model.
    
    Args:
        model: Neural network model
        X: Input data
        y: Target data
        eps: Small perturbation for gradient computation
        
    Returns:
        Dictionary of numerical gradients for all weights and biases
    """
    numerical_gradients = {}
    
    # Get a reference prediction and loss
    original_prediction = model.forward(X)
    original_loss, _ = model.loss_fn(original_prediction, y)
    
    # Keep track of parametrized layers
    param_layer_idx = 0
    
    # For each layer that has parameters
    for i, layer in enumerate(model.layers):
        if hasattr(layer, 'W') and hasattr(layer, 'b'):
            # Initialize gradient matrices
            numerical_dW = np.zeros_like(layer.W)
            numerical_db = np.zeros_like(layer.b)
            
            # Compute gradients for weights
            for r in range(layer.W.shape[0]):
                for c in range(layer.W.shape[1]):
                    # Add small perturbation
                    layer.W[r, c] += eps
                    
                    # Forward pass with perturbed weight
                    perturbed_pred = model.forward(X)
                    perturbed_loss, _ = model.loss_fn(perturbed_pred, y)
                    
                    # Compute numerical gradient
                    numerical_dW[r, c] = (perturbed_loss - original_loss) / eps
                    
                    # Restore original weight
                    layer.W[r, c] -= eps
            
            # Compute gradients for biases
            for b_idx in range(layer.b.shape[0]):
                # Add small perturbation
                layer.b[b_idx] += eps
                
                # Forward pass with perturbed bias
                perturbed_pred = model.forward(X)
                perturbed_loss, _ = model.loss_fn(perturbed_pred, y)
                
                # Compute numerical gradient
                numerical_db[b_idx] = (perturbed_loss - original_loss) / eps
                
                # Restore original bias
                layer.b[b_idx] -= eps
            
            # Store the computed gradients with their parameter layer index
            numerical_gradients[f'param_layer{param_layer_idx}_W'] = numerical_dW
            numerical_gradients[f'param_layer{param_layer_idx}_b'] = numerical_db
            
            # Increment parameter layer index
            param_layer_idx += 1
    
    return numerical_gradients

def get_analytical_gradients(model, X, y):
    """
    Get the analytical gradients computed by backpropagation.
    
    Args:
        model: Neural network model
        X: Input data
        y: Target data
        
    Returns:
        Dictionary of analytical gradients for all weights and biases
    """
    analytical_gradients = {}
    
    # Forward pass
    predictions = model.forward(X)
    
    # Get loss gradient
    _, grad = model.loss_fn(predictions, y)
    
    # Backward pass through the network
    for layer in reversed(model.layers):
        grad = layer.backward(grad)
    
    # Collect gradients from each layer with parameters
    param_layer_idx = 0
    for i, layer in enumerate(model.layers):
        if hasattr(layer, 'dW') and hasattr(layer, 'db'):
            analytical_gradients[f'param_layer{param_layer_idx}_W'] = layer.dW
            analytical_gradients[f'param_layer{param_layer_idx}_b'] = layer.db
            param_layer_idx += 1
    
    return analytical_gradients

def check_gradient_match(numerical, analytical, threshold=1e-5):
    """
    Check if numerical and analytical gradients match within a threshold.
    
    Args:
        numerical: Dictionary of numerical gradients
        analytical: Dictionary of analytical gradients
        threshold: Threshold for relative error
        
    Returns:
        Boolean indicating success and dictionary of relative errors
    """
    all_match = True
    relative_errors = {}
    
    # Debug print for keys
    print("Numerical gradient keys:", numerical.keys())
    print("Analytical gradient keys:", analytical.keys())
    
    for key in numerical.keys():
        if key not in analytical:
            print(f"Warning: Key {key} not found in analytical gradients")
            all_match = False
            continue
            
        # Compute relative error
        num = numerical[key]
        ana = analytical[key]
        
        # Avoid division by zero
        denominator = np.maximum(np.abs(num) + np.abs(ana), 1e-8)
        relative_error = np.max(np.abs(num - ana) / denominator)
        relative_errors[key] = relative_error
        
        if relative_error > threshold:
            all_match = False
    
    return all_match, relative_errors

if __name__ == "__main__":
    print("\n==== Gradient Check for Tiny Network (2-2-2) ====")
    
    # Create a tiny dataset
    np.random.seed(42)
    X = np.random.randn(10, 2)  # 10 samples, 2 features
    y = np.eye(2)[np.random.randint(0, 2, 10)]  # One-hot encoded targets
    
    # Create a tiny network (2 inputs, 2 hidden neurons, 2 outputs)
    layers = [
        Dense(2, 2),  # 2 inputs -> 2 hidden neurons
        ReLU(),
        Dense(2, 2),  # 2 hidden neurons -> 2 outputs
        Softmax()
    ]
    
    # Create optimizer
    optimizer = SGD(learning_rate=0.01)
    
    # Create model
    model = NeuralNet(layers, cross_entropy_loss, optimizer)
    
    # Compute numerical gradients
    print("Computing numerical gradients (this may take a moment)...")
    numerical_gradients = compute_numerical_gradient(model, X, y)
    
    # Compute analytical gradients
    print("Computing analytical gradients via backpropagation...")
    analytical_gradients = get_analytical_gradients(model, X, y)
    
    # Check if gradients match
    success, relative_errors = check_gradient_match(numerical_gradients, analytical_gradients)
    
    # Print results
    print("\n==== Gradient Check Results ====")
    for key, error in relative_errors.items():
        status = "✅ PASS" if error < 1e-5 else "❌ FAIL"
        print(f"{key}: Relative Error = {error:.8f} {status}")
    
    # Final result
    if success:
        print("\n✅ All gradient checks PASSED! The backpropagation is correctly implemented.")
    else:
        print("\n❌ Some gradient checks FAILED! The backpropagation may have issues.")
    
    print("\nThis confirms my manual check of numerical vs. analytical gradients.") 