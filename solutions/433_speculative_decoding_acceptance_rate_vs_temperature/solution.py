import numpy as np

def acceptance_rate_vs_temperature(
    draft_logits: np.ndarray, 
    target_logits: np.ndarray, 
    temperatures: np.ndarray) -> list:
    """
    Compute speculative decoding expected acceptance rate at various temperatures.
    
    Args:
        draft_logits: Logits from draft model, shape (vocab_size,)
        target_logits: Logits from target model, shape (vocab_size,)
        temperatures: Array of temperature values to evaluate
    
    Returns:
        List of acceptance rates (floats rounded to 4 decimal places)
    """
    rates = []
    for t in temperatures:
        tgt_swt = np.exp(target_logits/t)/np.sum(np.exp(target_logits/t))
        drft_swt = np.exp(draft_logits/t)/np.sum(np.exp(draft_logits/t))
        p = np.minimum(tgt_swt, drft_swt)
        rates.append(np.sum(p))
    return rates
    
