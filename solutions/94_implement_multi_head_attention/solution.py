import numpy as np
from typing import Tuple

def compute_qkv(X: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Query, Key, and Value matrices.
    
    Args:
        X: Input matrix of shape (seq_len, d_model)
        W_q, W_k, W_v: Weight matrices of shape (d_model, d_model)
    
    Returns:
        Q, K, V matrices each of shape (seq_len, d_model)
    """
    return X @ W_q, X @ W_k, X @ W_v

def self_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    """
    Compute scaled dot-product self-attention.
    
    Args:
        Q: Query matrix of shape (seq_len, d_k)
        K: Key matrix of shape (seq_len, d_k)
        V: Value matrix of shape (seq_len, d_k)
    
    Returns:
        Attention output of shape (seq_len, d_k)
    """
    scores = Q.dot(K.T)/np.sqrt(K.shape[1])
    softed = []
    for i in range(len(scores)):
        soft = np.exp(scores[i] - np.max(scores[i]))/np.sum(np.exp(scores[i] - np.max(scores[i])))
        softed.append(soft)
    
    return softed @ V

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, n_heads: int) -> np.ndarray:
    """
    Compute multi-head attention.
    
    Args:
        Q, K, V: Matrices of shape (seq_len, d_model)
        n_heads: Number of attention heads
    
    Returns:
        Attention output of shape (seq_len, d_model)
    """
    seq_len, d_model = Q.shape
    d_k = d_model // n_heads

    Q_d = np.reshape(Q, (seq_len, n_heads, d_k))
    K_d = np.reshape(K, (seq_len, n_heads, d_k))
    V_d = np.reshape(V, (seq_len, n_heads, d_k))

    Q_d = np.transpose(Q_d, (1, 0, 2))
    K_d = np.transpose(K_d, (1, 0, 2))
    V_d = np.transpose(V_d, (1, 0, 2))

    list_scores = []
    for h in range(n_heads):
        Q_h, K_h, V_h = Q_d[h], K_d[h], V_d[h]
        attention_scores = self_attention(Q_h, K_h, V_h)
        list_scores.append(attention_scores)
    list_scores = np.array(list_scores)
    list_scores = np.transpose(list_scores, (1, 0, 2))

    return np.reshape(list_scores, (seq_len, d_model))