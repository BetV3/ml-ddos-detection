import numpy as np

def one_hot_encode(y, num_classes=None):
    """
    Convert integer labels to one-hot encoded vectors.
    
    Args:
        y: Integer labels, shape (n_samples,)
        num_classes: Number of classes. If None, will be inferred from y.
        
    Returns:
        One-hot encoded labels, shape (n_samples, num_classes)
    """
    if num_classes is None:
        num_classes = np.max(y) + 1
    
    n_samples = y.shape[0]
    one_hot = np.zeros((n_samples, num_classes))
    one_hot[np.arange(n_samples), y] = 1
    return one_hot

def normalize(X, axis=0):
    """
    Normalize data to have zero mean and unit variance.
    
    Args:
        X: Input data, shape (n_samples, n_features)
        axis: Axis along which to normalize
        
    Returns:
        Normalized data, shape (n_samples, n_features)
    """
    mean = np.mean(X, axis=axis, keepdims=True)
    std = np.std(X, axis=axis, keepdims=True)
    return (X - mean) / (std + 1e-8)

def batch_iterator(X, y, batch_size, shuffle=True):
    """
    Create batches from data.
    
    Args:
        X: Input data, shape (n_samples, n_features)
        y: Target data, shape (n_samples, n_targets)
        batch_size: Size of each batch
        shuffle: Whether to shuffle the data
        
    Yields:
        Tuples of (X_batch, y_batch)
    """
    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    
    if shuffle:
        np.random.shuffle(indices)
    
    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch_indices = indices[start_idx:end_idx]
        
        yield X[batch_indices], y[batch_indices]

def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    """
    Convert input data to columns for efficient convolution operations.
    
    Args:
        input_data: Input data with shape (batch_size, channels, height, width)
        filter_h: Filter height
        filter_w: Filter width
        stride: Stride
        pad: Padding
        
    Returns:
        Columns with shape (batch_size, filter_h*filter_w*channels, output_height*output_width)
    """
    N, C, H, W = input_data.shape
    
    # Add padding if needed
    if pad > 0:
        padded_data = np.pad(input_data, ((0, 0), (0, 0), (pad, pad), (pad, pad)), 'constant')
    else:
        padded_data = input_data
    
    out_h = (H + 2*pad - filter_h) // stride + 1
    out_w = (W + 2*pad - filter_w) // stride + 1
    
    # Extract column data
    cols = np.zeros((N, C, filter_h, filter_w, out_h, out_w))
    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            cols[:, :, y, x, :, :] = padded_data[:, :, y:y_max:stride, x:x_max:stride]
    
    # Reshape to 2D array
    cols = cols.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
    
    return cols

def col2im(cols, input_shape, filter_h, filter_w, stride=1, pad=0):
    """
    Convert columns back to input data shape.
    
    Args:
        cols: Columns
        input_shape: Original input shape (batch_size, channels, height, width)
        filter_h: Filter height
        filter_w: Filter width
        stride: Stride
        pad: Padding
        
    Returns:
        Input data with original shape
    """
    N, C, H, W = input_shape
    out_h = (H + 2*pad - filter_h) // stride + 1
    out_w = (W + 2*pad - filter_w) // stride + 1
    
    # Reshape to 6D array
    cols = cols.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)
    
    # Initialize output array
    if pad > 0:
        img = np.zeros((N, C, H + 2*pad, W + 2*pad))
    else:
        img = np.zeros((N, C, H, W))
    
    # Add column values to the image
    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += cols[:, :, y, x, :, :]
    
    # Remove padding if needed
    if pad > 0:
        return img[:, :, pad:-pad, pad:-pad]
    return img 