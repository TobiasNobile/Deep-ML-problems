import numpy as np

def train(X, y, W, b):
    """
    Train linear regression weights on standardized data.
    
    Args:
        X: numpy array of shape (n_samples, n_features) -- standardized features
        y: numpy array of shape (n_samples,) -- standardized targets
        W: numpy array of shape (n_features,) -- initial random weights
        b: float -- initial bias (0.0)
    
    Returns:
        W: numpy array of shape (n_features,) -- trained weights
        b: float -- trained bias
    """
    lr = 0.01
    beta1, beta2 = 0.1, 0.2
    m, v = np.ones(W.shape), np.ones(W.shape)
    for i in range(1000):
        pred = X @ W + b
        gradient_w = 2/len(X)*X.T @ (X @ W + b - y)
        gradient_b = 2/len(X)*np.sum(X @ W + b - y)
        
        m = beta1 * m + (1 - beta1) * gradient_w
        v = beta2 * v + (1 - beta2) * gradient_w**2
        W = W - lr * m / (np.sqrt(v) + 1e-16)
        b = b - lr*gradient_b
        
    return W, b

