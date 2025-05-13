"""
Simple script to check if all imports work correctly.
"""
import sys
import os

# Add the parent directory to the path to ensure imports work
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("Attempting to import all modules...")

try:
    # Import layers
    from code.layers import Layer, Dense, Conv2D, MaxPool, Flatten, Dropout, BatchNorm
    print("Layers imported successfully")

    # Import activations
    from code.activations import ReLU, Sigmoid, Softmax, Tanh, LeakyReLU, ELU
    from code.activations import relu, sigmoid, tanh, softmax, leaky_relu
    print("Activations imported successfully")

    # Import network
    from code.network import NeuralNet
    print("Network imported successfully")

    # Import optimizers
    from code.optimizers import SGD, Adam
    print("Optimizers imported successfully")

    # Import losses
    from code.losses import mse_loss, cross_entropy_loss, binary_cross_entropy_loss, hinge_loss
    print("Losses imported successfully")

    # Import utils
    from code.utils import one_hot_encode, normalize, batch_iterator, im2col, col2im
    print("Utils imported successfully")

    print("\nAll modules imported successfully!")
    
except ImportError as e:
    print(f"Import error: {str(e)}")
except Exception as e:
    print(f"Error: {str(e)}") 