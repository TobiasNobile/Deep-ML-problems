import numpy as np

def classifier_free_guidance_step(
    eps_uncond: list,
    eps_cond: list,
    x_t: list,
    guidance_scale: float,
    alpha_bar_t: float,
    alpha_bar_t_minus_1: float,
    beta_t: float
) -> tuple:
    """
    Perform one reverse diffusion step with Classifier-Free Guidance.
    
    Args:
        eps_uncond: Unconditional noise prediction from the model
        eps_cond: Conditional noise prediction from the model
        x_t: Current noisy sample at timestep t
        guidance_scale: CFG weight (w). w=1 gives standard conditional, w>1 amplifies conditioning
        alpha_bar_t: Cumulative product of alpha up to timestep t
        alpha_bar_t_minus_1: Cumulative product of alpha up to timestep t-1
        beta_t: Noise schedule value at timestep t
    
    Returns:
        Tuple of (guided_eps, predicted_x0, posterior_mean) as lists rounded to 4 decimals
    """
    eps_uncond = np.array(eps_uncond)
    eps_cond = np.array(eps_cond)
    x_t = np.array(x_t)

    guided_eps = eps_uncond + guidance_scale*(eps_cond - eps_uncond)

    predicted_x0 = (x_t - np.sqrt(1-alpha_bar_t)*guided_eps)/np.sqrt(alpha_bar_t)

    coeff_x0 = np.sqrt(alpha_bar_t_minus_1)*beta_t/(1-alpha_bar_t)
    coeff_xt = np.sqrt(1 - beta_t)*(1 - alpha_bar_t_minus_1)/(1-alpha_bar_t)
    post_mean = coeff_x0*predicted_x0 + coeff_xt*x_t
    
    guided_eps = np.round(guided_eps, 3).tolist()
    predicted_x0 = np.round(predicted_x0, 3).tolist()
    post_mean = np.round(post_mean, 4).tolist()

    return guided_eps, predicted_x0, post_mean