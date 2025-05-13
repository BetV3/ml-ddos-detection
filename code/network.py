import numpy as np
from tqdm import tqdm

class NeuralNet:
    """
    Neural Network class that ties layers together and manages training.
    
    This is the core class of my neural network implementation, which 
    integrates the various layer components and handles the training process.
    """
    def __init__(self, layers, loss_fn, optimizer):
        """
        Initialize the neural network.
        
        I designed this to take a flexible list of layers, allowing for easy 
        creation of different network architectures.
        
        Args:
            layers: List of layer objects
            loss_fn: Loss function (from losses.py)
            optimizer: Optimizer object (from optimizers.py)
        """
        self.layers = layers
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        
    def forward(self, X):
        """
        Forward pass through all layers.
        
        I implemented this to sequentially pass the input through each layer,
        with each layer transforming the data and passing it to the next one.
        
        Args:
            X: Input data, shape (batch_size, input_dim)
            
        Returns:
            Output of the network
        """
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return output
    
    def backward(self, grad):
        """
        Backward pass through all layers.
        
        I implemented this to propagate gradients in the reverse order of the forward pass,
        which is how backpropagation works in neural networks.
        
        Args:
            grad: Gradient from the loss function
            
        Returns:
            Gradient with respect to input
        """
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad
    
    def get_params_and_grads(self):
        """
        Get parameters and gradients from all layers.
        
        I created this helper method to collect all trainable parameters and 
        their gradients to be used by the optimizer during training.
        
        Returns:
            List of (param, grad) tuples for optimization
        """
        params_and_grads = []
        for layer in self.layers:
            if hasattr(layer, 'W') and hasattr(layer, 'dW'):
                params_and_grads.append((layer.W, layer.dW))
            if hasattr(layer, 'b') and hasattr(layer, 'db'):
                params_and_grads.append((layer.b, layer.db))
        return params_and_grads
    
    def train(self, X_train, y_train, batch_size=32, epochs=10, X_val=None, y_val=None):
        """
        Train the neural network.
        
        I implemented mini-batch stochastic gradient descent with support for
        validation data to monitor the training progress and prevent overfitting.
        
        Args:
            X_train: Training data, shape (n_samples, input_dim)
            y_train: Training labels, shape (n_samples, output_dim)
            batch_size: Size of mini-batches
            epochs: Number of epochs to train for
            X_val: Validation data
            y_val: Validation labels
            
        Returns:
            History of training and validation losses
        """
        n_samples = X_train.shape[0]
        history = {'train_loss': [], 'val_loss': []}
        
        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
            
            epoch_loss = 0
            num_batches = int(np.ceil(n_samples / batch_size))
            
            # Mini-batch iterations
            for i in tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{epochs}"):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, n_samples)
                
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                
                # Forward pass
                predictions = self.forward(X_batch)
                
                # Compute loss and gradient
                loss, grad = self.loss_fn(predictions, y_batch)
                epoch_loss += loss
                
                # Backward pass
                self.backward(grad)
                
                # Update weights
                self.optimizer.update(self.get_params_and_grads())
            
            # Compute average loss for the epoch
            avg_loss = epoch_loss / num_batches
            history['train_loss'].append(avg_loss)
            
            # Validation if provided
            if X_val is not None and y_val is not None:
                val_predictions = self.forward(X_val)
                val_loss, _ = self.loss_fn(val_predictions, y_val)
                history['val_loss'].append(val_loss)
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}")
            else:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
                
        return history
    
    def predict(self, X):
        """
        Make predictions for input data.
        
        I designed this method to easily generate predictions on new data
        after the model has been trained.
        
        Args:
            X: Input data, shape (n_samples, input_dim)
            
        Returns:
            Predictions, shape (n_samples, output_dim)
        """
        return self.forward(X) 