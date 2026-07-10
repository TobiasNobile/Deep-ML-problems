import numpy as np

def jensen_shannon_divergence(P: list[float], Q: list[float]) -> float:
	"""
	Compute the Jensen-Shannon Divergence between two probability distributions.
	
	Args:
		P: First probability distribution
		Q: Second probability distribution
	
	Returns:
		Jensen-Shannon Divergence value
	"""
	P, Q = np.array(P), np.array(Q)
		
	M = 0.5 * (P + Q)
	D_p_m = np.sum([P[i]*np.log(P[i]/M[i]) if P[i] != 0 else 0  for i in range(len(P))])
	D_q_m = np.sum([Q[i]*np.log(Q[i]/M[i]) if Q[i] != 0 else 0 for i in range(len(P))])
	JSD = 1/2*(D_p_m + D_q_m)
	return JSD
