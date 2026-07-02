import numpy as np

def adaptive_layer_norm(x: np.ndarray, c: np.ndarray,
                        W_scale: np.ndarray, b_scale: np.ndarray,
                        W_shift: np.ndarray, b_shift: np.ndarray,
                        epsilon: float = 1e-5) -> np.ndarray:
    """
    Apply Adaptive Layer Normalization.
    
    Args:
        x: Input features of shape (B, D)
        c: Conditioning embeddings of shape (B, C)
        W_scale: Weight matrix for scale projection, shape (C, D)
        b_scale: Bias for scale projection, shape (D,)
        W_shift: Weight matrix for shift projection, shape (C, D)
        b_shift: Bias for shift projection, shape (D,)
        epsilon: Small constant for numerical stability
    
    Returns:
        Adaptively normalized output of shape (B, D)
    """
    scale = c @ W_scale + b_scale
    shift = c @ W_shift + b_shift
    x_mu = np.mean(x, axis=1, keepdims=True)
    x_var = np.var(x, axis=1, keepdims=True)
    x_norm = (x  - x_mu)/np.sqrt(x_var + epsilon)
    return scale * x_norm + shift