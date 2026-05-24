def rejection_sampling_best_of_k(candidates, scores):
    """
    Select the highest-scoring candidate per prompt.

    Args:
        candidates: list of N lists, each containing K candidate outputs.
        scores: list of N lists, each containing K reward scores.

    Returns:
        List of N selected candidates.
    """
    selected = []
    for iPrompt in range(len(candidates)):
        outputs = candidates[iPrompt]
        rewards = scores[iPrompt]
        selected.append(outputs[rewards.index(max(rewards))])
    return selected