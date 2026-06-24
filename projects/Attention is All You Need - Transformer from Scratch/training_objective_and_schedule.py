import torch

def shift_targets_right_with_start_token(target_ids, start_token_id):
    """
    Prepend start_token_id and drop the last column so output shape matches target_ids
    """
    B, L = target_ids.shape
    column = torch.full((B, 1), start_token_id)
    return torch.cat((column, target_ids[:, :-1]), dim=1)

def compute_noam_learning_rate(step, d_model, warmup_steps):
    """
    Return the Noam warmup learning rate for the given step.
    """
    return 1/d_model**(1/2)*min(1/step**(1/2), step*1/warmup_steps**(3/2))

def build_uniform_smoothing_distribution(shape, vocab_size, epsilon):
    """
    Return a float tensor of `shape` filled with epsilon / (vocab_size - 2).
    """
    return torch.full(shape, epsilon / (vocab_size - 2))

def set_confidence_on_gold_tokens(smoothed_distribution, gold_token_ids, confidence):
    """Place confidence mass at gold-token positions of a smoothed target distribution."""
    distrib = torch.clone(smoothed_distribution)
    gold = torch.unsqueeze(gold_token_ids, dim=2)
    return distrib.scatter_(2, gold, confidence)