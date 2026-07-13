import numpy as np

def train(X_train, y_train, X_val, y_val):
    """
    Train a binary classifier.
    
    Args:
        X_train: numpy array of shape (n_samples, 30) -- standardized features
        y_train: numpy array of shape (n_samples,) -- binary labels (0 or 1)
        X_val:   numpy array of shape (n_val, 30) -- standardized
        y_val:   numpy array of shape (n_val,) -- validation labels
    
    Returns:
        predict: callable that takes X (n, 30) and returns y_pred (n,) of 0s and 1s
    """
    W = np.random.randn(X_train.shape[1])
    for _ in range(1000):
        pred = X_train @ W
        pred = 1/(1+np.exp(-pred))
        loss = -1/len(X_train)*np.sum((y_train - pred)[:, None]*X_train, axis=0)
        W = W - 0.01*loss
    
    def predict(X):
        return np.where(X @ W > 0.5, 1, 0)
    return predict
