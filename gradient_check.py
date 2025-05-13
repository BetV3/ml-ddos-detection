"""
Gradient Checking Test for Neural Network Implementation

This script implements numerical gradient checking to validate the
analytical gradients computed by our backpropagation implementation.
It compares the analytically computed gradients with numerically
approximated gradients for different layers.
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from code.network import NeuralNet
from code.layers import Dense
from code.activations import ReLU, Sigmoid, Tanh
from code.losses import mse_loss
from code.optimizers import SGD

def compute_numerical_gradient(network, X, y, param, param_idx, layer_idx, epsilon=1e-7):
    """
    Compute numerical gradient for a specific parameter using the finite difference method.
    
    Args:
        network: Neural network model
        X: Input data
        y: Target data
        param: Parameter array to compute gradient for (weights or biases)
        param_idx: Index of the parameter in the flattened array
        layer_idx: Index of the layer that contains this parameter
        epsilon: Small perturbation for finite difference
        
    Returns:
        Numerical approximation of the gradient
    """
    # Create a flat copy of the parameter
    param_flat = param.flatten()
    
    # Store original value
    orig_val = param_flat[param_idx]
    
    # Forward pass with param + epsilon
    param_flat[param_idx] = orig_val + epsilon
    param_plus = param_flat.reshape(param.shape)
    
    # Update the correct layer's parameters
    if param is network.layers[layer_idx].W:
        network.layers[layer_idx].W = param_plus
    elif param is network.layers[layer_idx].b:
        network.layers[layer_idx].b = param_plus
    
    y_plus = network.forward(X)
    loss_plus, _ = mse_loss(y_plus, y)
    
    # Forward pass with param - epsilon
    param_flat[param_idx] = orig_val - epsilon
    param_minus = param_flat.reshape(param.shape)
    
    # Update the correct layer's parameters
    if param is network.layers[layer_idx].W:
        network.layers[layer_idx].W = param_minus
    elif param is network.layers[layer_idx].b:
        network.layers[layer_idx].b = param_minus
    
    y_minus = network.forward(X)
    loss_minus, _ = mse_loss(y_minus, y)
    
    # Compute numerical gradient using central difference
    numerical_grad = (loss_plus - loss_minus) / (2 * epsilon)
    
    # Restore original value
    param_flat[param_idx] = orig_val
    param_orig = param_flat.reshape(param.shape)
    
    # Restore the correct layer's parameters
    if param is network.layers[layer_idx].W:
        network.layers[layer_idx].W = param_orig
    elif param is network.layers[layer_idx].b:
        network.layers[layer_idx].b = param_orig
    
    return numerical_grad

def gradient_check_layer(network, X, y, layer_idx=0, num_checks=10, epsilon=1e-7, threshold=1e-5):
    """
    Check gradients for a specific layer by comparing analytical gradients with numerical approximations.
    
    Args:
        network: Neural network model
        X: Input data
        y: Target data
        layer_idx: Index of the layer to check
        num_checks: Number of parameters to check
        epsilon: Small perturbation for finite difference
        threshold: Threshold for relative error
        
    Returns:
        Boolean indicating if gradient check passed, and detailed results
    """
    print(f"Checking gradients for layer {layer_idx}...")
    
    # Get layer and its parameters
    layer = network.layers[layer_idx]
    
    if not hasattr(layer, 'W') or not hasattr(layer, 'b'):
        print(f"Layer {layer_idx} has no parameters to check")
        return True, []
    
    # Forward pass
    network.forward(X)
    
    # Backward pass
    _, grad = mse_loss(network.forward(X), y)
    network.backward(grad)
    
    # Get analytical gradients
    dW_analytical = layer.dW
    db_analytical = layer.db
    
    # Results storage
    results = []
    success = True
    
    # Check weights
    w_indices = np.random.choice(layer.W.size, min(num_checks, layer.W.size), replace=False)
    for idx in w_indices:
        w_flat_idx = idx
        w_idx = np.unravel_index(w_flat_idx, layer.W.shape)
        
        # Compute numerical gradient
        numerical_grad = compute_numerical_gradient(network, X, y, layer.W, w_flat_idx, layer_idx, epsilon)
        
        # Get analytical gradient
        analytical_grad = dW_analytical[w_idx]
        
        # Compute relative error
        if abs(numerical_grad) < 1e-10 and abs(analytical_grad) < 1e-10:
            rel_error = 0
        else:
            rel_error = abs(analytical_grad - numerical_grad) / max(abs(analytical_grad), abs(numerical_grad))
        
        passed = rel_error < threshold
        
        if not passed:
            success = False
        
        # Store result
        results.append({
            'param': 'W',
            'idx': w_idx,
            'numerical_grad': numerical_grad,
            'analytical_grad': analytical_grad,
            'rel_error': rel_error,
            'passed': passed
        })
        
        print(f"  W{w_idx}: numerical: {numerical_grad:.8f}, analytical: {analytical_grad:.8f}, "
              f"relative error: {rel_error:.8f} ({'PASSED' if passed else 'FAILED'})")
    
    # Check biases
    b_indices = np.random.choice(layer.b.size, min(num_checks, layer.b.size), replace=False)
    for idx in b_indices:
        b_flat_idx = idx
        b_idx = np.unravel_index(b_flat_idx, layer.b.shape)
        
        # Compute numerical gradient
        numerical_grad = compute_numerical_gradient(network, X, y, layer.b, b_flat_idx, layer_idx, epsilon)
        
        # Get analytical gradient
        analytical_grad = db_analytical[b_idx]
        
        # Compute relative error
        if abs(numerical_grad) < 1e-10 and abs(analytical_grad) < 1e-10:
            rel_error = 0
        else:
            rel_error = abs(analytical_grad - numerical_grad) / max(abs(analytical_grad), abs(numerical_grad))
        
        passed = rel_error < threshold
        
        if not passed:
            success = False
        
        # Store result
        results.append({
            'param': 'b',
            'idx': b_idx,
            'numerical_grad': numerical_grad,
            'analytical_grad': analytical_grad,
            'rel_error': rel_error,
            'passed': passed
        })
        
        print(f"  b{b_idx}: numerical: {numerical_grad:.8f}, analytical: {analytical_grad:.8f}, "
              f"relative error: {rel_error:.8f} ({'PASSED' if passed else 'FAILED'})")
    
    return success, results

def check_gradients_for_network(input_size=5, hidden_size=4, output_size=3, n_samples=10, 
                                activation=ReLU, epsilon=1e-7, threshold=1e-5):
    """
    Test gradient computations for a simple neural network.
    
    Args:
        input_size: Size of the input layer
        hidden_size: Size of the hidden layer
        output_size: Size of the output layer
        n_samples: Number of samples to use
        activation: Activation function to use
        epsilon: Small perturbation for finite difference
        threshold: Threshold for relative error
        
    Returns:
        Boolean indicating if all checks passed
    """
    print("\n==== Gradient Check for Neural Network ====")
    print(f"Network architecture: {input_size}->{hidden_size}->{output_size}")
    
    # Create a simple network
    layers = [
        Dense(input_size, hidden_size),
        activation(),
        Dense(hidden_size, output_size)
    ]
    
    network = NeuralNet(layers, mse_loss, SGD(learning_rate=0.01))
    
    # Generate random data
    np.random.seed(42)
    X = np.random.randn(n_samples, input_size)
    y = np.random.randn(n_samples, output_size)
    
    # Check gradients for the first layer
    success, results = gradient_check_layer(network, X, y, layer_idx=0)
    
    # Check gradients for the third layer (second Dense layer)
    success2, results2 = gradient_check_layer(network, X, y, layer_idx=2)
    
    overall_success = success and success2
    
    # Print overall result
    if overall_success:
        print("\n✅ All gradient checks PASSED!")
    else:
        print("\n❌ Some gradient checks FAILED!")
    
    return overall_success

def check_gradients_for_activations():
    """
    Test gradient computations for different activation functions.
    """
    print("\n==== Gradient Check for Different Activations ====")
    
    # Check ReLU
    print("\nChecking ReLU activation:")
    relu_success = check_gradients_for_network(activation=ReLU)
    
    # Check Sigmoid
    print("\nChecking Sigmoid activation:")
    sigmoid_success = check_gradients_for_network(activation=Sigmoid)
    
    # Check Tanh
    print("\nChecking Tanh activation:")
    tanh_success = check_gradients_for_network(activation=Tanh)
    
    return relu_success and sigmoid_success and tanh_success

if __name__ == "__main__":
    # Run gradient checks
    activation_success = check_gradients_for_activations()
    
    # Print overall result
    print("\n==== Gradient Check Summary ====")
    if activation_success:
        print("✅ All gradient checks PASSED! The backpropagation implementation is correct.")
    else:
        print("❌ Some gradient checks FAILED! The backpropagation implementation may have issues.") 