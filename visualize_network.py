"""
Neural Network Visualization

This script visualizes the structure of a neural network as a tree-like diagram,
showing the layers, connections, and dimensions of each component.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from code.network import NeuralNet
from code.layers import Dense, Conv2D, MaxPool, Flatten, BatchNorm, Dropout
from code.activations import ReLU, Sigmoid, Tanh, Softmax
from code.optimizers import Adam
from code.losses import cross_entropy_loss, mse_loss

def get_layer_info(layer, input_shape):
    """
    Get layer information including type, output shape, and parameters.
    
    Args:
        layer: Neural network layer
        input_shape: Input shape to the layer
        
    Returns:
        layer_type: Type of the layer
        output_shape: Shape of the output from this layer
        params: Number of trainable parameters
    """
    layer_type = layer.__class__.__name__
    params = 0
    
    # Forward pass to calculate output shape
    dummy_input = np.zeros((1,) + input_shape)
    output = layer.forward(dummy_input)
    output_shape = output.shape[1:]
    
    # Calculate number of parameters
    if hasattr(layer, 'W') and hasattr(layer, 'b'):
        params = np.prod(layer.W.shape) + np.prod(layer.b.shape)
    
    return layer_type, output_shape, params

def visualize_network(model, input_shape, filename="network_visualization.png"):
    """
    Visualize a neural network as a tree-like diagram.
    
    Args:
        model: Neural network model
        input_shape: Input shape (excluding batch dimension)
        filename: Output file name for the visualization
    """
    # Calculate appropriate figure height based on number of layers
    # More compact vertical spacing
    fig_height = max(8, (len(model.layers) + 1) * 1.2)
    
    plt.figure(figsize=(12, fig_height))
    
    # Set up plotting area with more space at the bottom
    ax = plt.gca()
    ax.set_xlim(0, 10)
    # Reduce vertical spacing
    ax.set_ylim(0, (len(model.layers) + 1) * 1.2 + 0.5)
    ax.axis('off')
    
    # Colors for different layer types
    colors = {
        'Dense': '#3498db',  # Blue
        'Conv2D': '#e74c3c',  # Red
        'MaxPool': '#2ecc71',  # Green
        'Flatten': '#f39c12',  # Orange
        'Dropout': '#9b59b6',  # Purple
        'BatchNorm': '#1abc9c',  # Turquoise
        'ReLU': '#95a5a6',  # Gray
        'Sigmoid': '#95a5a6',  # Gray
        'Tanh': '#95a5a6',  # Gray
        'Softmax': '#95a5a6'   # Gray
    }
    
    # Track current shape
    current_shape = input_shape
    
    # Calculate vertical positions more compactly
    total_height = (len(model.layers) + 1) * 1.2
    input_y_pos = total_height - 0.6  # Position input closer to first layer
    
    # Draw input layer
    plt.text(5, input_y_pos, f"Input: {input_shape}", 
             ha='center', va='center', fontsize=12, fontweight='bold',
             bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))
    
    # Draw each layer
    for i, layer in enumerate(model.layers):
        y_pos = total_height - (i + 1) * 1.2
        
        # Get layer info
        layer_type, output_shape, params = get_layer_info(layer, current_shape)
        current_shape = output_shape
        
        # Draw layer box
        box_width = 6
        box_height = 0.8
        box_x = 5 - box_width / 2
        box_y = y_pos - box_height / 2
        
        # Get color
        color = colors.get(layer_type, '#34495e')
        
        # Create box
        rect = Rectangle((box_x, box_y), box_width, box_height, facecolor=color, alpha=0.7, edgecolor='black')
        ax.add_patch(rect)
        
        # Add layer text
        layer_text = f"{layer_type}"
        plt.text(5, y_pos, layer_text, ha='center', va='center', fontsize=11, fontweight='bold', color='white')
        
        # Add shape and params text
        shape_text = f"Output: {output_shape}"
        param_text = f"Params: {params:,}" if params > 0 else ""
        plt.text(5, y_pos - 0.25, shape_text, ha='center', va='center', fontsize=9, color='white')
        if param_text:
            plt.text(5, y_pos + 0.25, param_text, ha='center', va='center', fontsize=9, color='white')
        
        # Add arrow connecting to previous layer or input
        if i == 0:
            # Connect to input
            arrow = FancyArrowPatch((5, input_y_pos - 0.3), (5, y_pos + 0.4), 
                                    arrowstyle='-|>', mutation_scale=20, 
                                    color='black', linewidth=1.5)
        else:
            # Connect to previous layer
            prev_y = y_pos + 1.2
            arrow = FancyArrowPatch((5, prev_y - 0.4), (5, y_pos + 0.4), 
                                    arrowstyle='-|>', mutation_scale=20, 
                                    color='black', linewidth=1.5)
        ax.add_patch(arrow)
    
    # Add total parameters
    total_params = sum(np.prod(layer.W.shape) + np.prod(layer.b.shape) 
                      for layer in model.layers if hasattr(layer, 'W') and hasattr(layer, 'b'))
    
    # Add total parameters as overlay on the last layer
    if len(model.layers) > 0:
        plt.text(5, total_height - len(model.layers) * 1.2 - 0.5, f"Total Parameters: {total_params:,}", 
                ha='center', va='center', fontsize=10, fontweight='bold', 
                bbox=dict(facecolor='#ecf0f1', edgecolor='black', boxstyle='round,pad=0.3'))
    
    # Add title
    plt.title("Neural Network Architecture", fontsize=16, fontweight='bold', pad=10)
    
    # Save the visualization with more padding
    plt.tight_layout()
    plt.savefig(filename, dpi=200, bbox_inches='tight', pad_inches=0.3)
    print(f"Network visualization saved as '{filename}'")
    plt.close()

def visualize_sample_networks():
    """
    Visualize several sample neural network architectures.
    """
    # 1. Simple classification network
    print("Visualizing simple classification network...")
    layers_simple = [
        Dense(10, 64),
        ReLU(),
        Dense(64, 32),
        ReLU(),
        Dense(32, 3),
        Softmax()
    ]
    model_simple = NeuralNet(layers_simple, cross_entropy_loss, Adam(learning_rate=0.01))
    visualize_network(model_simple, (10,), "simple_classification_network.png")
    
    # 2. Conv2D network for image classification
    print("Visualizing CNN for image classification...")
    layers_cnn = [
        Conv2D(input_channels=3, num_filters=16, kernel_size=3, stride=1, padding=1),
        ReLU(),
        MaxPool(pool_size=2, stride=2),
        Conv2D(input_channels=16, num_filters=32, kernel_size=3, stride=1, padding=1),
        ReLU(),
        MaxPool(pool_size=2, stride=2),
        Flatten(),
        Dense(32*7*7, 128),
        ReLU(),
        Dense(128, 10),
        Softmax()
    ]
    model_cnn = NeuralNet(layers_cnn, cross_entropy_loss, Adam(learning_rate=0.01))
    visualize_network(model_cnn, (3, 28, 28), "cnn_network.png")
    
    # 3. Regression network with dropout and batch normalization
    print("Visualizing regression network with regularization...")
    layers_reg = [
        Dense(8, 64),
        BatchNorm(64),
        ReLU(),
        Dropout(p=0.3),
        Dense(64, 32),
        BatchNorm(32),
        ReLU(),
        Dropout(p=0.3),
        Dense(32, 1)
    ]
    model_reg = NeuralNet(layers_reg, mse_loss, Adam(learning_rate=0.01))
    visualize_network(model_reg, (8,), "regression_network.png")
    
    print("All visualizations completed!")

if __name__ == "__main__":
    visualize_sample_networks() 