"""
Comprehensive Neural Network Testing Suite

This script runs all tests and validations for the neural network implementation:
1. Gradient checking tests
2. Unit tests for layers and activations
3. Model comparison with scikit-learn
4. Visualizations and benchmark tests

Use this script to verify the correctness of the implementation.
"""

import os
import sys
import subprocess
import numpy as np
import time
from sklearn.datasets import make_moons, make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from code.network import NeuralNet
from code.layers import Dense, BatchNorm, Dropout
from code.activations import ReLU, Sigmoid, Tanh
from code.optimizers import SGD, Adam
from code.losses import mse_loss, cross_entropy_loss
from code.utils import one_hot_encode


def print_header(title):
    """Print a formatted header for test sections."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def run_gradient_check():
    """Run gradient checking tests to validate backpropagation."""
    print_header("GRADIENT CHECKING TESTS")
    
    try:
        import gradient_check
        success = gradient_check.check_gradients_for_activations()
        return success
    except ImportError:
        print("❌ gradient_check.py not found! Please create the file first.")
        return False


def test_activations():
    """Test activation functions and their gradients."""
    print_header("ACTIVATION FUNCTIONS TESTS")
    
    # Create random input data
    np.random.seed(42)
    x = np.random.randn(5, 3)
    
    # Test ReLU
    print("\nTesting ReLU:")
    from code.activations import ReLU, relu, relu_derivative
    relu_layer = ReLU()
    out = relu_layer.forward(x)
    assert np.allclose(out, relu(x)), "ReLU forward pass failed"
    
    # Test ReLU backward
    grad_out = np.random.randn(*out.shape)
    grad_in = relu_layer.backward(grad_out)
    expected_grad = grad_out * relu_derivative(x)
    assert np.allclose(grad_in, expected_grad), "ReLU backward pass failed"
    print("✅ ReLU tests passed")
    
    # Test Sigmoid
    print("\nTesting Sigmoid:")
    from code.activations import Sigmoid, sigmoid, sigmoid_derivative
    sigmoid_layer = Sigmoid()
    out = sigmoid_layer.forward(x)
    assert np.allclose(out, sigmoid(x)), "Sigmoid forward pass failed"
    
    # Test Sigmoid backward
    grad_out = np.random.randn(*out.shape)
    grad_in = sigmoid_layer.backward(grad_out)
    expected_grad = grad_out * sigmoid_derivative(x)
    assert np.allclose(grad_in, expected_grad), "Sigmoid backward pass failed"
    print("✅ Sigmoid tests passed")
    
    # Test Tanh
    print("\nTesting Tanh:")
    from code.activations import Tanh, tanh, tanh_derivative
    tanh_layer = Tanh()
    out = tanh_layer.forward(x)
    assert np.allclose(out, tanh(x)), "Tanh forward pass failed"
    
    # Test Tanh backward
    grad_out = np.random.randn(*out.shape)
    grad_in = tanh_layer.backward(grad_out)
    expected_grad = grad_out * tanh_derivative(x)
    assert np.allclose(grad_in, expected_grad), "Tanh backward pass failed"
    print("✅ Tanh tests passed")
    
    return True


def test_layer_shapes():
    """Test that layers produce correct output shapes."""
    print_header("LAYER SHAPE TESTS")
    
    # Create random input data
    np.random.seed(42)
    batch_size = 10
    input_dim = 5
    hidden_dim = 7
    output_dim = 3
    
    x = np.random.randn(batch_size, input_dim)
    
    # Test Dense layer shapes
    print("\nTesting Dense layer shapes:")
    dense = Dense(input_dim, hidden_dim)
    out = dense.forward(x)
    assert out.shape == (batch_size, hidden_dim), f"Expected shape {(batch_size, hidden_dim)}, got {out.shape}"
    
    grad_out = np.random.randn(*out.shape)
    grad_in = dense.backward(grad_out)
    assert grad_in.shape == x.shape, f"Expected gradient shape {x.shape}, got {grad_in.shape}"
    print("✅ Dense layer shape tests passed")
    
    # Test BatchNorm layer shapes
    print("\nTesting BatchNorm layer shapes:")
    batch_norm = BatchNorm(hidden_dim)
    bn_out = batch_norm.forward(out)
    assert bn_out.shape == out.shape, f"Expected shape {out.shape}, got {bn_out.shape}"
    
    bn_grad_out = np.random.randn(*bn_out.shape)
    bn_grad_in = batch_norm.backward(bn_grad_out)
    assert bn_grad_in.shape == out.shape, f"Expected gradient shape {out.shape}, got {bn_grad_in.shape}"
    print("✅ BatchNorm layer shape tests passed")
    
    # Test Dropout layer shapes
    print("\nTesting Dropout layer shapes:")
    dropout = Dropout(p=0.5)
    dropout_out = dropout.forward(out)
    assert dropout_out.shape == out.shape, f"Expected shape {out.shape}, got {dropout_out.shape}"
    
    dropout_grad_out = np.random.randn(*dropout_out.shape)
    dropout_grad_in = dropout.backward(dropout_grad_out)
    assert dropout_grad_in.shape == out.shape, f"Expected gradient shape {out.shape}, got {dropout_grad_in.shape}"
    print("✅ Dropout layer shape tests passed")
    
    return True


def test_simple_network_training():
    """Test end-to-end training of a simple network."""
    print_header("SIMPLE NETWORK TRAINING TEST")
    
    # Create a simple binary classification dataset
    np.random.seed(42)
    X, y = make_classification(n_samples=100, n_features=5, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Convert to one-hot encoding
    y_train_one_hot = one_hot_encode(y_train, 2)
    
    # Create a simple network
    layers = [
        Dense(5, 8),
        ReLU(),
        Dense(8, 2),
        Sigmoid()
    ]
    
    # Create optimizer and network
    optimizer = SGD(learning_rate=0.1)
    model = NeuralNet(layers, cross_entropy_loss, optimizer)
    
    # Train the model
    print("Training a simple network...")
    history = model.train(X_train, y_train_one_hot, batch_size=32, epochs=10)
    
    # Evaluate model
    predictions = model.predict(X_test)
    pred_classes = np.argmax(predictions, axis=1)
    accuracy = accuracy_score(y_test, pred_classes)
    
    print(f"Test accuracy: {accuracy:.4f}")
    if accuracy > 0.6:
        print("✅ Simple network training test passed")
        return True
    else:
        print("❌ Simple network training test failed (accuracy too low)")
        return False


def run_external_tests():
    """Run the external test scripts."""
    print_header("EXTERNAL TEST SCRIPTS")
    
    test_files = [
        "test_network.py",
        "test_simple.py",
        "two_moons_test.py",
        "time_series_prediction.py"
    ]
    
    results = []
    
    for test_file in test_files:
        if os.path.exists(test_file):
            print(f"\nRunning {test_file}...")
            try:
                result = subprocess.run([sys.executable, test_file], 
                                        capture_output=True, text=True)
                if result.returncode == 0:
                    print(f"✅ {test_file} passed")
                    results.append(True)
                else:
                    print(f"❌ {test_file} failed with error:")
                    print(result.stderr)
                    results.append(False)
            except Exception as e:
                print(f"❌ Error running {test_file}: {e}")
                results.append(False)
        else:
            print(f"⚠️ {test_file} not found, skipping...")
    
    # Check if at least some tests ran and passed
    if results and all(results):
        print("\n✅ All external tests passed!")
        return True
    elif not results:
        print("\n⚠️ No external tests were found or run.")
        return False
    else:
        print("\n❌ Some external tests failed.")
        return False


def run_model_comparison():
    """Run model comparison with scikit-learn if the script exists."""
    print_header("MODEL COMPARISON WITH SCIKIT-LEARN")
    
    if os.path.exists("model_comparison.py"):
        print("Running model comparison...")
        try:
            result = subprocess.run([sys.executable, "model_comparison.py"], 
                                    capture_output=True, text=True)
            if result.returncode == 0:
                print("✅ Model comparison completed successfully")
                return True
            else:
                print("❌ Model comparison failed with error:")
                print(result.stderr)
                return False
        except Exception as e:
            print(f"❌ Error running model comparison: {e}")
            return False
    else:
        print("⚠️ model_comparison.py not found, skipping...")
        return False


def run_all_tests():
    """Run all tests and report overall success."""
    print_header("NEURAL NETWORK COMPREHENSIVE TEST SUITE")
    start_time = time.time()
    
    # Run all tests
    results = {
        "Gradient Check": run_gradient_check(),
        "Activations": test_activations(),
        "Layer Shapes": test_layer_shapes(),
        "Simple Network": test_simple_network_training(),
        "External Tests": run_external_tests(),
        "Model Comparison": run_model_comparison()
    }
    
    # Print summary
    print_header("TEST SUMMARY")
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name.ljust(20)}: {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n🎉 ALL TESTS PASSED! The neural network implementation is correct.")
    else:
        print("\n⚠️ SOME TESTS FAILED. Please review the issues above.")
    
    elapsed_time = time.time() - start_time
    print(f"\nTesting completed in {elapsed_time:.2f} seconds.")
    
    return all_passed


if __name__ == "__main__":
    run_all_tests() 