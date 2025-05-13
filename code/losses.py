import numpy as np

def mse_loss(predictions, targets):
    """
    Mean Squared Error loss function.
    
    Calculates the average squared difference between predictions and targets.
    MSE = (1/n) * sum((predictions - targets)^2)
    
    Often used for regression problems where the output is a continuous value.
    
    Args:
        predictions: Model predictions, shape (batch_size, output_dim)
        targets: Ground truth values, shape (batch_size, output_dim)
        
    Returns:
        tuple: (loss scalar, gradient with respect to predictions)
    """
    loss = np.mean((predictions - targets) ** 2)
    grad = 2 * (predictions - targets) / targets.size
    return loss, grad


def cross_entropy_loss(predictions, targets):
    """
    Cross Entropy loss function.
    
    Measures the performance of a classification model whose output
    is a probability distribution over classes.
    
    Commonly used for multi-class classification problems.
    
    Args:
        predictions: Model predictions (probability distribution), shape (batch_size, num_classes)
        targets: Ground truth labels (one-hot encoded or class indices), shape (batch_size, num_classes) or (batch_size,)
        
    Returns:
        tuple: (loss scalar, gradient with respect to predictions)
    """
    eps = 1e-12  # Small epsilon to avoid log(0)
    clipped_preds = np.clip(predictions, eps, 1 - eps)
    
    if targets.ndim == 1:
        # Convert targets to one-hot encoding if they're class indices
        n_samples = targets.shape[0]
        one_hot = np.zeros((n_samples, clipped_preds.shape[1]))
        one_hot[np.arange(n_samples), targets] = 1
        targets = one_hot
    
    # Compute negative log likelihood
    loss = -np.sum(targets * np.log(clipped_preds)) / targets.shape[0]
    
    # Gradient of cross entropy with respect to predictions
    grad = -(targets / clipped_preds) / targets.shape[0]
    
    return loss, grad


def binary_cross_entropy_loss(predictions, targets):
    """
    Binary Cross Entropy loss function.
    
    A special case of cross entropy for binary classification problems.
    BCE = -1/N * sum(targets * log(predictions) + (1-targets) * log(1-predictions))
    
    Args:
        predictions: Model predictions (probability of positive class), shape (batch_size, 1) or (batch_size, 2)
        targets: Ground truth labels (0 or 1), shape (batch_size, 1) or (batch_size, 2)
        
    Returns:
        tuple: (loss scalar, gradient with respect to predictions)
    """
    eps = 1e-12  # Small epsilon to avoid log(0)
    clipped_preds = np.clip(predictions, eps, 1 - eps)
    
    # Compute negative log likelihood
    loss = -np.mean(targets * np.log(clipped_preds) + (1 - targets) * np.log(1 - clipped_preds))
    
    # Gradient is (p-y)/(p*(1-p)) where p is prediction and y is target
    grad = (clipped_preds - targets) / (clipped_preds * (1 - clipped_preds))
    grad = grad / targets.size
    
    return loss, grad


def hinge_loss(predictions, targets):
    """
    Hinge loss for margin-based classification.
    
    Commonly used for SVM classification, penalizing incorrect predictions
    based on how far they are from the decision boundary.
    
    Args:
        predictions: Model predictions, shape (batch_size, 1)
        targets: Ground truth labels (-1 or 1), shape (batch_size, 1)
        
    Returns:
        tuple: (loss scalar, gradient with respect to predictions)
    """
    margin = 1 - predictions * targets
    loss = np.mean(np.maximum(0, margin))
    
    # Gradient: -y if margin > 0, otherwise 0
    grad = np.zeros_like(predictions)
    mask = margin > 0
    grad[mask] = -targets[mask] / targets.size
    
    return loss, grad
