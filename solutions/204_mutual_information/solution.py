import numpy as np

def mutual_information(joint_prob: list[list[float]]) -> float:
	"""
	Compute the mutual information between two random variables.
	
	Args:
		joint_prob: 2D joint probability distribution P(X,Y)
	
	Returns:
		Mutual information I(X;Y)
	"""
	p_x = np.sum(joint_prob, axis=1)
	p_y = np.sum(joint_prob, axis=0)
	p_x_y = np.outer(p_x, p_y)
	joint_prob = np.array(joint_prob)
	info = np.where(joint_prob > 0, joint_prob*np.log(joint_prob/p_x_y), 0)
	return np.sum(info)
