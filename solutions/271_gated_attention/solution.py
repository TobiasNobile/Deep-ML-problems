import numpy as np

def gated_attention(
    X: np.ndarray,
    W_q: np.ndarray,
    W_k: np.ndarray,
    W_v: np.ndarray,
    W_g: np.ndarray
) -> np.ndarray:
    """
    Compute Gated Attention output.
    
    Args:
        X: Input tensor of shape (seq_len, d_model)
        W_q: Query projection of shape (d_model, d_k)
        W_k: Key projection of shape (d_model, d_k)
        W_v: Value projection of shape (d_model, d_v)
        W_g: Gate projection of shape (d_model, d_v)
    
    Returns:
        Gated attention output of shape (seq_len, d_v), rounded to 4 decimal places
    
    Hint: First compute standard scaled dot-product attention, then apply
    a sigmoid gate to modulate the output.
    """
    Q, K, V = X @ W_q, X @ W_k, X @ W_v
    Y = Q @ K.T / np.sqrt(K.shape[1])
    for i in range(len(Y)):
        Y[i] = np.exp(Y[i])/sum(np.exp(Y[i]))
    Y = Y @ V
    Y = np.multiply(1/(1+np.exp(-(X @ W_g))), Y)
    return Y
