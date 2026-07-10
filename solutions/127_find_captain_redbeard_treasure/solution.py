def find_treasure(start_x: float) -> float:
    """
    Find the x-coordinate where f(x) = x^4 - 3x^3 + 2 is minimized.

  Returns:
        float: The x-coordinate of the minimum point.
    """
    def f_prime(x):
      return 4*x**3 - 9*x**2
    x = start_x
    v = 0
    for _ in range(100):
      v = 0.9 * v - 0.01*f_prime(x)
      x = x + v
    return x