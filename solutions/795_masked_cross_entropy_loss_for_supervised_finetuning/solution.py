import numpy as np

def masked_ce_loss(logits: np.ndarray, targets: np.ndarray, mask: np.ndarray) -> float:
    """
    Compute mean cross-entropy loss over masked (response) positions only.

    Args:
        logits: (seq_len, vocab_size) array of unnormalized scores.
        targets: (seq_len,) array of integer target token ids.
        mask: (seq_len,) boolean array; True = include in loss.

    Returns:
        Mean cross-entropy over positions where mask is True (float).
    """
    if mask.sum() == 0:
        return 0.0
    seq_len, vocab_size = logits.shape

    cross_entropy = 0
    for i in range(seq_len):
        if mask[i]:
            target_id = targets[i]
            token_entropy = -logits[i, target_id] + np.log(np.sum(np.exp(logits[i])))
            cross_entropy += token_entropy

    return cross_entropy/mask.sum()