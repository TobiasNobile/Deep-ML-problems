import numpy as np

def multivariate_kl_divergence(mu_p: np.ndarray, Cov_p: np.ndarray, mu_q: np.ndarray, Cov_q: np.ndarray) -> float:
    """
    Computes the KL divergence between two multivariate Gaussian distributions.
    
    Parameters:
    mu_p: mean vector of the first distribution
    Cov_p: covariance matrix of the first distribution
    mu_q: mean vector of the second distribution
    Cov_q: covariance matrix of the second distribution

    Returns:
    KL divergence as a float
    """
    const = np.log(np.linalg.det(Cov_q)/np.linalg.det(Cov_p))
    return 0.5*(const - Cov_p.shape[1]+ (mu_p - mu_q).T @ np.linalg.inv(Cov_q) @ (mu_p - mu_q) + np.trace(np.linalg.inv(Cov_q) @ Cov_p))