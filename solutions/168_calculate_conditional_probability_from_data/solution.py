def conditional_probability(data, x, y):
    """
    Returns the probability P(Y=y|X=x) from list of (X, Y) pairs.
    Args:
      data: List of (X, Y) tuples
      x: value of X to condition on
      y: value of Y to check
    Returns:
      float: conditional probability, rounded to 4 decimal places
    """
    subset = [t for t in data if t[0] == x]
    if not subset:
      return 0
    subset_y = [t for t in subset if t[1] == y]
    return len(subset_y)/len(subset)