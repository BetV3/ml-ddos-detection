import numpy as np

class ReLU:
    """
    Rectified Linear Unit activation function.
    
    This activation function is defined as f(x) = max(0, x), setting all negative
    values to zero and keeping positive values unchanged.
    
    Commonly used in hidden layers of neural networks for its computational
    efficiency and to help with the vanishing gradient problem.
    """
    def forward(self, x):
        """
        Forward pass for ReLU activation.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor with ReLU activation applied
        """
        self.mask = (x > 0)
        return x * self.mask

    def backward(self, grad_output):
        """
        Backward pass for ReLU activation.
        
        Args:
            grad_output: Gradient from the next layer
            
        Returns:
            Gradient with respect to the input
        """
        return grad_output * self.mask


class Sigmoid:
    """
    Sigmoid activation function.
    
    This activation function squashes input values to the range (0, 1)
    using the formula f(x) = 1 / (1 + e^(-x)).
    
    Often used in binary classification problems or for gates in recurrent networks.
    """
    def forward(self, x):
        """
        Forward pass for Sigmoid activation.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor with Sigmoid activation applied
        """
        self.out = 1 / (1 + np.exp(-x))
        return self.out

    def backward(self, grad_output):
        """
        Backward pass for Sigmoid activation.
        
        Args:
            grad_output: Gradient from the next layer
            
        Returns:
            Gradient with respect to the input
        """
        return grad_output * (self.out * (1 - self.out))


class Softmax:
    """
    Softmax activation function.
    
    This activation function converts a vector of real numbers into a probability
    distribution using the formula f(x_i) = e^(x_i) / sum(e^(x_j)) for all j.
    
    Often used in the output layer of multi-class classification problems.
    """
    def forward(self, x):
        """
        Forward pass for Softmax activation.
        
        Args:
            x: Input tensor, shape (batch_size, num_classes)
            
        Returns:
            Output tensor with Softmax activation applied
        """
        # Subtract max for numerical stability
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        self.out = exps / np.sum(exps, axis=1, keepdims=True)
        return self.out

    def backward(self, grad_output):
        """
        Backward pass for Softmax activation.
        
        Args:
            grad_output: Gradient from the next layer
            
        Returns:
            Gradient with respect to the input
        """
        batch, classes = grad_output.shape
        grad_input = np.empty_like(grad_output)
        # Softmax has a full Jacobian matrix, so we compute it for each sample
        for i in range(batch):
            y = self.out[i].reshape(-1, 1)
            jacobian = np.diagflat(y) - y.dot(y.T)
            grad_input[i] = jacobian.dot(grad_output[i])
        return grad_input


class Tanh:
    """
    Hyperbolic Tangent activation function.
    
    This activation function squashes input values to the range (-1, 1)
    using the formula f(x) = tanh(x).
    
    Often used in hidden layers as an alternative to ReLU, especially in
    recurrent neural networks.
    """
    def forward(self, x):
        """
        Forward pass for Tanh activation.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor with Tanh activation applied
        """
        self.out = np.tanh(x)
        return self.out
    
    def backward(self, grad_output):
        """
        Backward pass for Tanh activation.
        
        Args:
            grad_output: Gradient from the next layer
            
        Returns:
            Gradient with respect to the input
        """
        return grad_output * (1 - self.out**2)


class LeakyReLU:
    """
    Leaky Rectified Linear Unit activation function.
    
    This activation is a variant of ReLU that allows small negative values with
    a slope of alpha instead of setting them to zero: f(x) = max(alpha*x, x).
    
    Helps prevent "dying ReLU" problem where neurons can become inactive and
    stop learning.
    """
    def __init__(self, alpha=0.01):
        """
        Initialize LeakyReLU with specified negative slope.
        
        Args:
            alpha: Slope for negative inputs (default: 0.01)
        """
        self.alpha = alpha
        
    def forward(self, x):
        """
        Forward pass for LeakyReLU activation.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor with LeakyReLU activation applied
        """
        self.x = x
        return np.where(x >= 0, x, self.alpha * x)
    
    def backward(self, grad_output):
        """
        Backward pass for LeakyReLU activation.
        
        Args:
            grad_output: Gradient from the next layer
            
        Returns:
            Gradient with respect to the input
        """
        return np.where(self.x >= 0, grad_output, self.alpha * grad_output)


class ELU:
    """
    Exponential Linear Unit activation function.
    
    This activation is defined as f(x) = x if x > 0 and alpha * (e^x - 1) if x <= 0.
    
    Unlike ReLU, ELU can produce negative outputs and is smooth at x=0,
    which can improve learning dynamics.
    """
    def __init__(self, alpha=1.0):
        """
        Initialize ELU with specified alpha parameter.
        
        Args:
            alpha: Scale for the negative factor (default: 1.0)
        """
        self.alpha = alpha
        
    def forward(self, x):
        """
        Forward pass for ELU activation.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor with ELU activation applied
        """
        self.x = x
        return np.where(x >= 0, x, self.alpha * (np.exp(x) - 1))
    
    def backward(self, grad_output):
        """
        Backward pass for ELU activation.
        
        Args:
            grad_output: Gradient from the next layer
            
        Returns:
            Gradient with respect to the input
        """
        return np.where(self.x >= 0, grad_output, grad_output * self.alpha * np.exp(self.x))


# Standalone activation functions and their derivatives

def relu(x):
    """
    Rectified Linear Unit function.
    
    Args:
        x: Input tensor
        
    Returns:
        Output tensor with ReLU applied
    """
    return np.maximum(0, x)

def relu_derivative(x):
    """
    Derivative of ReLU function.
    
    Args:
        x: Input tensor
        
    Returns:
        Derivative of ReLU at x
    """
    return np.where(x > 0, 1, 0)

def sigmoid(x):
    """
    Sigmoid function.
    
    Args:
        x: Input tensor
        
    Returns:
        Output tensor with sigmoid applied
    """
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    """
    Derivative of sigmoid function.
    
    Args:
        x: Input tensor
        
    Returns:
        Derivative of sigmoid at x
    """
    s = sigmoid(x)
    return s * (1 - s)

def tanh(x):
    """
    Hyperbolic tangent function.
    
    Args:
        x: Input tensor
        
    Returns:
        Output tensor with tanh applied
    """
    return np.tanh(x)

def tanh_derivative(x):
    """
    Derivative of tanh function.
    
    Args:
        x: Input tensor
        
    Returns:
        Derivative of tanh at x
    """
    return 1 - np.tanh(x)**2

def softmax(x):
    """
    Softmax function.
    
    Args:
        x: Input tensor of shape (batch_size, num_classes)
        
    Returns:
        Probability distribution over classes
    """
    exps = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)

def leaky_relu(x, alpha=0.01):
    """
    Leaky ReLU function.
    
    Args:
        x: Input tensor
        alpha: Slope for negative inputs (default: 0.01)
        
    Returns:
        Output tensor with leaky ReLU applied
    """
    return np.where(x >= 0, x, alpha * x)

def leaky_relu_derivative(x, alpha=0.01):
    """
    Derivative of leaky ReLU function.
    
    Args:
        x: Input tensor
        alpha: Slope for negative inputs (default: 0.01)
        
    Returns:
        Derivative of leaky ReLU at x
    """
    return np.where(x >= 0, 1, alpha)
