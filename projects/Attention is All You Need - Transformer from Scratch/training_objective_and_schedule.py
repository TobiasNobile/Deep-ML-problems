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

def zero_pad_column_and_pad_token_rows(smoothed_distribution, gold_token_ids, pad_id):
    """
    Zero the pad column and the rows where the gold token equals pad_id
    """
    distrib = torch.clone(smoothed_distribution)
    distrib[...,  pad_id] = 0
    mask = pad_id == gold_token_ids
    distrib[mask] = 0
    return distrib

def compute_label_smoothed_kl_loss(log_probabilities, smoothed_distribution):
    """Return the summed KL loss over all (batch, time, vocab) entries."""
    return -torch.sum(log_probabilities * smoothed_distribution) + 0.0 # + 0.0 for IEEE convention absorbing negative zero

def average_loss_over_non_pad_tokens(total_loss, gold_token_ids, pad_id):
    """
    Divide total_loss by the count of non-pad tokens in gold_token_ids
    """
    n = torch.sum(gold_token_ids != pad_id)
    if n == 0:
        return total_loss
    return total_loss/n

def compute_token_accuracy_ignoring_pad(log_probabilities, gold_token_ids, pad_id):
    """
    Argmax over vocab, compare to gold, average over non-pad positions only
    """
    p = torch.argmax(log_probabilities, 2)
    numerator = (p==gold_token_ids) * (gold_token_ids!= pad_id)
    if torch.sum(gold_token_ids!= pad_id) == 0:
        return torch.Tensor([0])
    return torch.sum(numerator) / torch.sum(gold_token_ids!= pad_id)
