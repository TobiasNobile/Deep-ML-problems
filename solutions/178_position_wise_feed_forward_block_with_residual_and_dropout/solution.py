import numpy as np

def ffn(
	x: list[float], 
	W1: list[list[float]], 
	b1: list[float], 
	W2: list[list[float]], 
	b2: list[float], 
	dropout_p: float=0.1, 
	seed: int=42) -> list[float]:
	"""
	Implement a position-wise feed-forward block with residual and dropout.

	Args:
		x: input vector
		W1, b1: first linear layer parameters
		W2, b2: second linear layer parameters
		dropout_p: dropout probability
		seed: random seed for reproducibility

	Returns:
		Output vector after FFN block (rounded to 4 decimals)
	"""
	x, W1, b1, W2, b2 = np.array(x), np.array(W1), np.array(b1), np.array(W2), np.array(b2)
	h1 = W1 @ x + b1
	z1 = np.maximum(0, h1)
	h2 = W2 @ z1 + b2
	rng = np.random.RandomState(seed)
	keep = rng.rand(*h2.shape) >= dropout_p
	h2 = h2 * keep / (1.0 - dropout_p)
	return np.round(h2 + x, 4).tolist()
