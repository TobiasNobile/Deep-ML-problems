import numpy as np

def sliding_window_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, window_size: int) -> np.ndarray:
    """
    Compute sliding window attention.
    
    Args:
        Q: Query matrix of shape (seq_len, d_k)
        K: Key matrix of shape (seq_len, d_k)
        V: Value matrix of shape (seq_len, d_v)
        window_size: Number of positions to the left and right each query can attend to
    
    Returns:
        Output matrix of shape (seq_len, d_v), rounded to 4 decimal places.
    """
    attention = Q @ K.T / np.sqrt(K.shape[1])
    mask = np.zeros((len(Q), len(Q)))
    for i in range(len(mask)):
        for j in range(len(mask)):
            mask[i, j] = float('-inf') if abs(i-j) > window_size else 0
    attention += mask
    for i in range(len(attention)):
        attention[i] = np.exp(attention[i])/np.sum(np.exp(attention[i]))
    return attention @ V
