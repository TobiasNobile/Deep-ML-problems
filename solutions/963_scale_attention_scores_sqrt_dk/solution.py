import numpy as np

def scaled_attention_weights(Q: np.ndarray, K: np.ndarray) -> list:
    """
    Compute scaled dot-product attention weights.

    Args:
        Q: (n_q, d_k) query matrix
        K: (n_k, d_k) key matrix

    Returns:
        Attention weights of shape (n_q, n_k) as a nested list,
        each entry rounded to 4 decimal places.
    """
    scores = Q @ K.T
    scores /= np.sqrt(K.shape[1])
    softed = np.exp(scores - np.max(scores, axis=1, keepdims=True))/ np.sum(np.exp(scores - np.max(scores, axis=1, keepdims=True)), axis=1, keepdims=True)
    return softed