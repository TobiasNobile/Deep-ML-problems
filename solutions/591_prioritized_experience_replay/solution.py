import numpy as np

def prioritized_replay_sample(priorities: list, batch_size: int, alpha: float = 0.6, beta: float = 0.4, seed: int = 42) -> dict:
	"""
	Sample a batch from a replay buffer using prioritized experience replay.

	Args:
		priorities: list of priority values for each experience (positive floats)
		batch_size: number of experiences to sample
		alpha: prioritization exponent (0 = uniform, 1 = full prioritization)
		beta: importance sampling exponent (0 = no correction, 1 = full correction)
		seed: random seed for reproducibility

	Returns:
		dict with 'indices', 'probabilities', and 'weights'
	"""
	N = len(priorities)
	sampling_prob = np.array(priorities)**alpha/np.sum(np.array(priorities)**alpha)
	indices = list(range(N))
	np.random.seed(seed)
	sample = np.random.choice(indices, batch_size, False, sampling_prob)
	weights = (sampling_prob[sample]*N)**(-beta)
	weights /= np.max(weights)
	sampling_prob = np.round(sampling_prob, 4)
	return {
		'indices': sample, 
		'probabilities': sampling_prob.tolist(),
		'weights': weights.tolist()
	}
