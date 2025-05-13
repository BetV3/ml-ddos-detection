import numpy as np
try:
    from code.utils import im2col, col2im
except ImportError:
    from .utils import im2col, col2im

class Layer:
    """
    Base class for all layers in the neural network.
    
    This abstract class defines the interface that all layer implementations
    must follow with forward and backward methods for the forward and
    backward passes of backpropagation.
    """
    def forward(self, input):
        """
        Forward pass through the layer.
        
        Args:
            input: Input data to the layer
            
        Returns:
            Output data from the layer
        """
        raise NotImplementedError

    def backward(self, grad_output):
        """
        Backward pass through the layer.
        
        Args:
            grad_output: Gradient from the next layer
            
        Returns:
            Gradient with respect to the input of this layer
        """
        raise NotImplementedError


class Dense(Layer):
    """
    Fully connected (dense) layer.
    
    This layer implements a fully connected layer where each neuron is
    connected to all neurons in the previous layer.
    
    Attributes:
        W: Weight matrix of shape (input_size, output_size)
        b: Bias vector of shape (output_size,)
        dW: Gradient of weights (computed during backward pass)
        db: Gradient of biases (computed during backward pass)
    """
    def __init__(self, input_size, output_size):
        """
        Initialize the dense layer with Xavier/Glorot initialization.
        
        Args:
            input_size: Number of input features
            output_size: Number of output features
        """
        # Xavier/Glorot initialization for better gradient flow
        limit = np.sqrt(6.0 / (input_size + output_size))
        self.W = np.random.uniform(-limit, limit, (input_size, output_size))
        self.b = np.zeros(output_size)

    def forward(self, input):
        """
        Forward pass for the dense layer.
        
        Args:
            input: Input data, shape (batch_size, input_size)
            
        Returns:
            Output data, shape (batch_size, output_size)
        """
        self.input = input
        return input.dot(self.W) + self.b

    def backward(self, grad_output):
        """
        Backward pass for the dense layer.
        
        Args:
            grad_output: Gradient from the next layer, shape (batch_size, output_size)
            
        Returns:
            Gradient with respect to the input, shape (batch_size, input_size)
        
        Side effects:
            Sets self.dW and self.db with gradients for weight update
        """
        self.dW = self.input.T.dot(grad_output)
        self.db = grad_output.sum(axis=0)
        return grad_output.dot(self.W.T)


class Conv2D(Layer):
    """
    2D Convolutional layer.
    
    This layer applies convolution operations on the input tensor.
    
    Attributes:
        W: Weights/filters of shape (num_filters, input_channels, kernel_size, kernel_size)
        b: Biases of shape (num_filters,)
        stride: Stride of the convolution
        padding: Padding to apply to the input
    """
    def __init__(self, input_channels, num_filters, kernel_size, stride=1, padding=0):
        """
        Initialize the convolutional layer.
        
        Args:
            input_channels: Number of input channels
            num_filters: Number of filters (output channels)
            kernel_size: Size of the convolutional kernel
            stride: Stride for the convolution
            padding: Zero-padding to add to the input
        """
        self.input_channels = input_channels
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # He initialization (better for ReLU)
        scale = np.sqrt(2.0 / (input_channels * kernel_size * kernel_size))
        self.W = np.random.normal(0, scale, (num_filters, input_channels, kernel_size, kernel_size))
        self.b = np.zeros(num_filters)

    def forward(self, input):
        """
        Forward pass of 2D convolution using im2col for efficiency.
        
        Args:
            input: Input tensor of shape (batch_size, input_channels, height, width)
            
        Returns:
            Output tensor of shape (batch_size, num_filters, output_height, output_width)
        """
        self.input = input
        batch_size, C, H, W = input.shape
        KH, KW = self.kernel_size, self.kernel_size
        out_h = (H + 2*self.padding - KH) // self.stride + 1
        out_w = (W + 2*self.padding - KW) // self.stride + 1
        
        # Reshape filters for matrix multiplication
        W_col = self.W.reshape(self.num_filters, -1)
        
        # Transform input to column format for efficient convolution
        x_col = im2col(input, KH, KW, self.stride, self.padding)
        self.x_col = x_col
        
        # Perform convolution as matrix multiplication
        out = W_col.dot(x_col.T) + self.b.reshape(-1, 1)
        
        # Reshape output
        out = out.reshape(self.num_filters, out_h, out_w, batch_size)
        out = out.transpose(3, 0, 1, 2)
        
        return out

    def backward(self, grad_output):
        """
        Backward pass of 2D convolution using im2col for efficiency.
        
        Args:
            grad_output: Gradient tensor from next layer of shape 
                         (batch_size, num_filters, output_height, output_width)
                         
        Returns:
            Gradient with respect to input of shape (batch_size, input_channels, height, width)
            
        Side effects:
            Sets self.dW and self.db with gradients for weight update
        """
        batch_size, C, H, W = self.input.shape
        KH, KW = self.kernel_size, self.kernel_size
        
        # Reshape gradient for convolution operation
        grad_output_reshaped = grad_output.transpose(1, 2, 3, 0).reshape(self.num_filters, -1)
        
        # Compute gradient w.r.t. weights
        self.dW = grad_output_reshaped.dot(self.x_col).reshape(self.W.shape)
        
        # Compute gradient w.r.t. bias
        self.db = np.sum(grad_output, axis=(0, 2, 3))
        
        # Compute gradient w.r.t. input
        W_reshape = self.W.reshape(self.num_filters, -1)
        dx_col = W_reshape.T.dot(grad_output_reshaped)
        
        # Convert column format back to input format
        dx = col2im(dx_col.T, self.input.shape, KH, KW, self.stride, self.padding)
        
        return dx


class MaxPool(Layer):
    """
    Max Pooling layer.
    
    This layer performs max pooling on the input tensor, reducing spatial dimensions
    by selecting the maximum value in each pooling window.
    
    Attributes:
        pool_size: Size of the pooling window
        stride: Stride of the pooling operation
    """
    def __init__(self, pool_size=2, stride=2):
        """
        Initialize the max pooling layer.
        
        Args:
            pool_size: Size of the pooling window
            stride: Stride for the pooling operation
        """
        self.pool_size = pool_size
        self.stride = stride

    def forward(self, input):
        """
        Forward pass for max pooling.
        
        Args:
            input: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Output tensor of shape (batch_size, channels, output_height, output_width)
        """
        self.input = input
        batch, C, H, W = input.shape
        ph, pw = self.pool_size, self.pool_size
        out_h = (H - ph) // self.stride + 1
        out_w = (W - pw) // self.stride + 1
        output = np.zeros((batch, C, out_h, out_w))
        self.mask = np.zeros_like(input)
        
        # Manual implementation of max pooling
        for n in range(batch):
            for c in range(C):
                for i in range(out_h):
                    for j in range(out_w):
                        h_start, h_end = i*self.stride, i*self.stride+ph
                        w_start, w_end = j*self.stride, j*self.stride+pw
                        region = input[n, c, h_start:h_end, w_start:w_end]
                        max_val = np.max(region)
                        output[n, c, i, j] = max_val
                        # Create mask to track which position had the max value
                        self.mask[n, c, h_start:h_end, w_start:w_end] += (region == max_val)
                        
        return output

    def backward(self, grad_output):
        """
        Backward pass for max pooling.
        
        Args:
            grad_output: Gradient tensor from next layer of shape 
                         (batch_size, channels, output_height, output_width)
                         
        Returns:
            Gradient with respect to input of shape (batch_size, channels, height, width)
        """
        batch, C, H, W = self.input.shape
        ph, pw = self.pool_size, self.pool_size
        out_h, out_w = grad_output.shape[2], grad_output.shape[3]
        grad_input = np.zeros_like(self.input)
        
        # Route the gradient only through the max values identified during forward pass
        for n in range(batch):
            for c in range(C):
                for i in range(out_h):
                    for j in range(out_w):
                        h_start, h_end = i*self.stride, i*self.stride+ph
                        w_start, w_end = j*self.stride, j*self.stride+pw
                        mask_region = self.mask[n, c, h_start:h_end, w_start:w_end]
                        grad_input[n, c, h_start:h_end, w_start:w_end] += grad_output[n, c, i, j] * mask_region
                        
        return grad_input


class Flatten(Layer):
    """
    Flatten layer.
    
    This layer flattens the input tensor (except for the batch dimension)
    to prepare it for dense layers.
    """
    def forward(self, input):
        """
        Forward pass for flattening.
        
        Args:
            input: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Flattened tensor of shape (batch_size, channels*height*width)
        """
        self.input_shape = input.shape
        batch_size = input.shape[0]
        return input.reshape(batch_size, -1)

    def backward(self, grad_output):
        """
        Backward pass for flattening.
        
        Args:
            grad_output: Gradient tensor of shape (batch_size, flattened_dim)
            
        Returns:
            Reshaped gradient of original shape (batch_size, channels, height, width)
        """
        return grad_output.reshape(self.input_shape)


class Dropout(Layer):
    """
    Dropout layer for regularization.
    
    This layer randomly sets a fraction of inputs to zero during training
    to prevent overfitting.
    
    Attributes:
        p: Probability of dropping a unit
        mask: Binary mask used during forward pass
    """
    def __init__(self, p=0.5):
        """
        Initialize the dropout layer.
        
        Args:
            p: Probability of dropping a unit (between 0 and 1)
        """
        self.p = p
        self.mask = None
        
    def forward(self, input, training=True):
        """
        Forward pass for dropout.
        
        Args:
            input: Input tensor
            training: Whether in training mode (dropout is applied) or inference mode
            
        Returns:
            Output tensor with dropout applied during training
        """
        if training:
            # Create binary mask and scale by 1/(1-p) to maintain expected value
            self.mask = np.random.binomial(1, 1-self.p, size=input.shape) / (1-self.p)
            return input * self.mask
        else:
            # No dropout during inference
            return input
            
    def backward(self, grad_output):
        """
        Backward pass for dropout.
        
        Args:
            grad_output: Gradient tensor from next layer
            
        Returns:
            Gradient with respect to input with dropout applied
        """
        # Apply the same mask to the gradient
        return grad_output * self.mask


class BatchNorm(Layer):
    """
    Batch Normalization layer.
    
    This layer normalizes the activations of the previous layer to 
    have zero mean and unit variance, then applies a learnable scale and shift.
    
    Attributes:
        gamma: Scale parameter (learnable)
        beta: Shift parameter (learnable)
        eps: Small constant for numerical stability
        momentum: Momentum for running statistics
        running_mean: Running average of batch means for inference
        running_var: Running average of batch variances for inference
    """
    def __init__(self, num_features, eps=1e-5, momentum=0.9):
        """
        Initialize the batch normalization layer.
        
        Args:
            num_features: Number of features/channels
            eps: Small constant for numerical stability
            momentum: Momentum for running statistics updates
        """
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        # Learnable parameters
        self.gamma = np.ones(num_features)
        self.beta = np.zeros(num_features)
        # Running statistics for inference
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)
        
    def forward(self, input, training=True):
        """
        Forward pass for batch normalization.
        
        Args:
            input: Input tensor of shape (batch_size, num_features)
            training: Whether in training mode or inference mode
            
        Returns:
            Normalized and scaled output tensor
        """
        if training:
            # Calculate mean and variance for this batch
            self.input = input
            batch_size = input.shape[0]
            self.sample_mean = np.mean(input, axis=0)
            self.sample_var = np.var(input, axis=0)
            
            # Normalize
            self.x_norm = (input - self.sample_mean) / np.sqrt(self.sample_var + self.eps)
            out = self.gamma * self.x_norm + self.beta
            
            # Update running statistics for inference
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * self.sample_mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * self.sample_var
            
            return out
        else:
            # Use running statistics for inference
            x_norm = (input - self.running_mean) / np.sqrt(self.running_var + self.eps)
            return self.gamma * x_norm + self.beta
        
    def backward(self, grad_output):
        """
        Backward pass for batch normalization.
        
        Args:
            grad_output: Gradient tensor from next layer of shape (batch_size, num_features)
            
        Returns:
            Gradient with respect to input of same shape
            
        Side effects:
            Sets self.dgamma and self.dbeta gradients for parameter updates
        """
        batch_size = self.input.shape[0]
        
        # Gradient with respect to gamma and beta
        self.dgamma = np.sum(grad_output * self.x_norm, axis=0)
        self.dbeta = np.sum(grad_output, axis=0)
        
        # Gradient with respect to normalized input
        dx_norm = grad_output * self.gamma
        
        # Gradient with respect to variance
        dvar = np.sum(dx_norm * (self.input - self.sample_mean) * -0.5 * 
                     np.power(self.sample_var + self.eps, -1.5), axis=0)
        
        # Gradient with respect to mean
        dmean = np.sum(dx_norm * -1 / np.sqrt(self.sample_var + self.eps), axis=0)
        dmean += dvar * np.mean(-2 * (self.input - self.sample_mean), axis=0)
        
        # Gradient with respect to input
        dx = dx_norm / np.sqrt(self.sample_var + self.eps)
        dx += dvar * 2 * (self.input - self.sample_mean) / batch_size
        dx += dmean / batch_size
        
        return dx