import numpy as np

def train_logreg(X: np.ndarray, y: np.ndarray, learning_rate: float, iterations: int) -> tuple[list[float], ...]:
    """
    Gradient-descent training algorithm for logistic regression, optimizing
    parameters with Binary Cross Entropy loss.

    Notes:
        - Initialize all coefficients to zero
        - Add bias column (ones) as the FIRST column of X
        - Use the BCE loss summed over samples (not averaged):
          -sum(y*log(p) + (1-y)*log(1-p))
    """
    y = np.reshape(y, (len(y), 1))
    X = np.insert(X, 0, np.ones(len(X)), axis=1)
    W = np.zeros((X.shape[1], 1))
    losses = []

    for _ in range(iterations):
      pred = X @ W
      p = 1/(1+np.exp(-pred))
      loss = -np.sum(y*np.log(p) + (1-y)*np.log(1-p))
      losses.append(loss)

      W = W - learning_rate*(X.T @ (p - y))
    
    return np.round(W.flatten(), 4).tolist(), np.round(losses, 4)
