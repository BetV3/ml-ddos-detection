import numpy as np

class SGD:
    """
    Stochastic Gradient Descent optimizer.
    """
    def __init__(self, learning_rate=0.01, momentum=0.0):
        """
        Initialize SGD optimizer.
        
        Args:
            learning_rate: Learning rate
            momentum: Momentum factor (0 for no momentum)
        """
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity = {}
    
    def update(self, params_and_grads):
        """
        Update parameters using gradients.
        
        Args:
            params_and_grads: List of (param, grad) tuples
        """
        for i, (param, grad) in enumerate(params_and_grads):
            # Initialize velocity if it doesn't exist
            if i not in self.velocity:
                self.velocity[i] = np.zeros_like(param)
            
            # Update velocity with momentum
            self.velocity[i] = self.momentum * self.velocity[i] - self.learning_rate * grad
            
            # Update parameters
            param += self.velocity[i]


class Adam:
    """
    Adam optimizer (Adaptive Moment Estimation).
    """
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """
        Initialize Adam optimizer.
        
        Args:
            learning_rate: Learning rate
            beta1: Exponential decay rate for first moment estimates
            beta2: Exponential decay rate for second moment estimates
            epsilon: Small constant for numerical stability
        """
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}  # First moment estimates
        self.v = {}  # Second moment estimates
        self.t = 0   # Timestep
    
    def update(self, params_and_grads):
        """
        Update parameters using gradients.
        
        Args:
            params_and_grads: List of (param, grad) tuples
        """
        self.t += 1
        
        for i, (param, grad) in enumerate(params_and_grads):
            # Initialize moments if they don't exist
            if i not in self.m:
                self.m[i] = np.zeros_like(param)
                self.v[i] = np.zeros_like(param)
            
            # Update biased first moment estimate
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            
            # Update biased second raw moment estimate
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)
            
            # Compute bias-corrected first moment estimate
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            
            # Compute bias-corrected second raw moment estimate
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            
            # Update parameters
            param -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon) 