import math

def beta_distribution_stats(x: float, alpha: float, beta_param: float) -> dict:
    """
    Compute Beta distribution statistics.
    
    Args:
        x: Value at which to evaluate the PDF
        alpha: First shape parameter (alpha > 0)
        beta_param: Second shape parameter (beta > 0)
    
    Returns:
        Dictionary with 'pdf', 'mean', and 'variance'
    """
    beta = math.gamma(alpha)*math.gamma(beta_param) / math.gamma(alpha + beta_param)
    pdf = x**(alpha -1 ) * (1-x)**(beta_param - 1) / beta if 0 < x < 1 else 0
    mean = alpha / (alpha + beta_param)
    var = alpha*beta_param / ((alpha + beta_param)**2*(alpha + beta_param +1 ))
    return {
        'pdf': pdf, 
        'mean': mean, 
        'variance': var
    }