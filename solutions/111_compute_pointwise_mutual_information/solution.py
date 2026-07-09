import numpy as np

def compute_pmi(joint_counts, total_counts_x, total_counts_y, total_samples):
	joint_prob = joint_counts / total_samples
	x_prob, y_prob = total_counts_x/total_samples, total_counts_y / total_samples
	return np.log2(joint_prob/(x_prob*y_prob))