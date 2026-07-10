import numpy as np

def cross_doc_attention_mask(token_ids: list[int], doc_ids: list[int]) -> np.ndarray:
    """
    Build a boolean attention mask for a packed sequence of multiple documents.

    Args:
        token_ids: packed token ids (length N)
        doc_ids:   document id per token (length N)

    Returns:
        Boolean array of shape (N, N). True means the row (query) token can
        attend to the column (key) token.
    """
    N = len(token_ids)
    mask = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if doc_ids[i] == doc_ids[j] and j <= i:
                mask[i][j] = 1
    return mask
