import numpy as np

def guidance_attention_mask(
    chunk_sizes: list[int],
    current_chunk: int
) -> np.ndarray:
    """
    Build a boolean attention mask for chunked autoregressive video generation.

    Args:
        chunk_sizes:   tokens per chunk, in order
        current_chunk: index of the chunk being generated

    Returns:
        Boolean array of shape (total_tokens, total_tokens).
        True means the row token can attend to the column token.
    """
    history_size = sum(chunk_sizes[:current_chunk])
    current_size = chunk_sizes[current_chunk]

    history_tokens = np.ones((history_size+current_size, history_size))
    low_til = np.tril(np.ones((current_size, current_size)))
    top_right = np.zeros((history_size, current_size))
    right = np.concatenate((top_right, low_til), axis=0)

    return np.concatenate((history_tokens, right), axis=1)
