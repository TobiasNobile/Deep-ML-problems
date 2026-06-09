import numpy as np

def layer_normalization(X: np.ndarray, gamma: np.ndarray, beta: np.ndarray, epsilon: float = 1e-5) -> np.ndarray:
	"""
	Perform Layer Normalization.
	"""
	batch_size = X.shape[0]
	seq_len = X.shape[1]
	feature_dim = X.shape[2]

	output = np.random.randn(batch_size, seq_len, feature_dim)
	for i in range(batch_size):
		for j in range(seq_len):
			feature_mean = np.mean(X[i, j])
			feature_variance = np.var(X[i, j])
			output[i, j] = gamma*(X[i, j] - feature_mean)/np.sqrt(feature_variance + epsilon) + beta
	return output
