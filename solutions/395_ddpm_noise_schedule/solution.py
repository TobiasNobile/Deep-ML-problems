import numpy as np

def ddpm_linear_schedule(beta_start: float, beta_end: float, T: int) -> dict:
    """
    Compute the DDPM linear noise schedule and derived quantities.

    Args:
        beta_start: Starting value of the beta (noise variance) schedule.
        beta_end: Ending value of the beta schedule.
        T: Number of diffusion timesteps.

    Returns:
        Dictionary with keys: 'betas', 'alphas', 'alpha_bars',
        'sqrt_alpha_bars', 'sqrt_one_minus_alpha_bars'.
    """
    schedule = {
        'betas': [], 
        'alphas': [], 
        'alpha_bars': [], 
        'sqrt_alpha_bars': [], 'sqrt_one_minus_alpha_bars': []
    }
    if T > 1:
        beta_t = beta_start + np.array([(t-1)/(T-1)*(beta_end - beta_start) for t in range(1, T+1)])
    else:
        beta_t = [beta_start]
    
    for b in beta_t:
        b = float(b)
        schedule['betas'].append(b)
        schedule['alphas'].append(1-b)
        schedule['alpha_bars'].append(np.prod(schedule['alphas']))
        schedule['sqrt_alpha_bars'].append(float(np.sqrt(schedule['alpha_bars'][-1])))
        schedule['sqrt_one_minus_alpha_bars'].append(float(np.sqrt(1 - schedule['alpha_bars'][-1])))
    return schedule
        