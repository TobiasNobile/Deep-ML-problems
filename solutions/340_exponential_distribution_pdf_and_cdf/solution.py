import numpy as np

def exponential_distribution(x: list, lam: float) -> dict:
    """
    Compute exponential distribution properties.
    
    Args:
        x: Points at which to evaluate PDF and CDF
        lam: Rate parameter (lambda) of the distribution
        
    Returns:
        Dictionary with 'pdf', 'cdf', 'mean', and 'variance' keys
    """
    x = np.array(x)
    pdf = np.round(lam*np.exp(-lam*x), 4)
    cdf = np.round(1 - np.exp(-lam*x), 4)
    return {
        "pdf": np.where(x >= 0, pdf, 0).tolist() if lam > 0 else None,
        "cdf": np.where(x >= 0, cdf, 0).tolist() if lam > 0 else None,
        "mean" : np.round(1/lam, 4).tolist() if lam > 0 else None,
        "variance": np.round(1/lam**2, 4).tolist() if lam > 0 else None
    }