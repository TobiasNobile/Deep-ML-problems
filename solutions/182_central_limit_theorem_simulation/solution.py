import numpy as np

def simulate_clt(distribution: str, n: int, runs: int = 10000, seed: int = 42) -> dict:
    """
    Simulate the Central Limit Theorem.

    Args:
        distribution (str): The distribution to sample from ('uniform', 'exponential', 'bernoulli').
        n (int): Sample size.
        runs (int): Number of repeated experiments.
        seed (int): Random seed for reproducibility.

    Returns:
        dict: {'mean': float, 'std': float} of the standardized sample means.
    """
    np.random.seed(seed)
    z = []
    for i in range(runs):
        match distribution:
            case 'uniform':
                sample = np.random.uniform(0.0, 1.0, n)
                x_bar = np.mean(sample)
                x_bar = (x_bar - 0.5) / (1/ np.sqrt(12*n))
            case 'exponential':
                sample = np.random.exponential(1.0, n)
                x_bar = np.mean(sample)
                x_bar = (x_bar - 1)*np.sqrt(n)
            case 'bernoulli':
                sample = np.random.uniform(0.0, 1.0, n)
                sample_b = np.where(sample < 0.3, 1, 0)
                x_bar = np.mean(sample_b)
                x_bar = (x_bar - 0.3) / np.sqrt(0.21/n)
        z.append(x_bar)
    z = np.array(z)
    return {
        "mean": np.mean(z).item(),
        "std": np.std(z).item()
    }