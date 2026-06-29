def conditional_probability(joint_distribution: dict) -> float:
	"""
	Compute conditional probability P(A|B) from a joint probability distribution.

	Args:
		joint_distribution (dict): dictionary with keys ('A','B'), ('A','`B'), ('`A','B'), ('`A','`B')

	Returns:
		float: Conditional probability P(A|B)
	"""
	P_b = joint_distribution[('A','B')] + joint_distribution[('`A','B')]
	P_a_inter_b = joint_distribution[('A','B')]
	return P_a_inter_b / P_b 