import numpy as np

def logit_normal_timesteps(num_samples: int, mean: float = 0.0, std: float = 1.0, seed: int = 42) -> np.ndarray:
	"""
	Sample diffusion timesteps from a logit-normal distribution.
	
	Parameters:
		num_samples: Number of timestep samples to generate
		mean: Mean of the underlying normal distribution
		std: Standard deviation of the underlying normal distribution
		seed: Random seed for reproducibility
	
	Returns:
		numpy array of shape (num_samples,) with values in (0, 1)
	"""
	np.random.seed(seed)
	sample_norm = np.random.normal(mean, std, num_samples)
	return 1/(1+np.exp(-sample_norm))