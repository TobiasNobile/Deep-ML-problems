from collections import Counter

def calculate_f1_score(y_true, y_pred):
	"""
	Calculate the F1 score based on true and predicted labels.

	Args:
		y_true (list): True labels (ground truth).
		y_pred (list): Predicted labels.

	Returns:
		float: The F1 score rounded to three decimal places.
	"""
	data = [(true, pred) for true, pred in zip(y_true, y_pred)]
	c = Counter(data)

	precision = c[(1, 1)]/(c[(1, 1)] + c[(0, 1)])
	recall = c[(1, 1)]/(c[(1, 1)] + c[(1, 0)]) if c[(1, 1)] + c[(1, 0)] != 0 else 0.0

	f1 = 2*precision*recall/(precision+recall) if precision + recall != 0 else 0.0
		
	return round(f1,3)