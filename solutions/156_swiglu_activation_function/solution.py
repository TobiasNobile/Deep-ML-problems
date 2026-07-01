import numpy as np

def SwiGLU(x: np.ndarray) -> np.ndarray:
    """
    Args:
        x: np.ndarray of shape (batch_size, 2d)
    Returns:
        np.ndarray of shape (batch_size, d)
    """
    d = x.shape[1]
    x1, x2 = x[:, :d//2], x[:, d//2:]
    scores = x1 * x2 * 1/(1+np.exp(-x2))
    return scores