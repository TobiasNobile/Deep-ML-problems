import numpy as np

def entropy_and_cross_entropy(P: list[float], Q: list[float]) -> tuple[float, float]:
	"""
	Compute entropy of P and cross-entropy between P and Q.
	
	Args:
		P: True probability distribution
		Q: Predicted probability distribution
	
	Returns:
		Tuple of (entropy H(P), cross-entropy H(P,Q))
	"""
	if 0 in P:
		entropy = 0.0
	else:
		entropy = -sum([p*np.log(p) for p in P])
	cross_entropy = -sum([P[i]*np.log(Q[i]) for i in range(len(P))])
	return entropy, cross_entropy