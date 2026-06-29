import math
from statistics import NormalDist

def cdf(x):
    return 1/2*(1+ math.erf(x/math.sqrt(2)))

def calculate_power(effect_size: float, sample_size_per_group: int, alpha: float = 0.05, two_tailed: bool = True) -> float:
    """
    Calculate statistical power for a two-sample z-test.
    
    Parameters:
    effect_size: Cohen's d (standardized effect size)
    sample_size_per_group: Number of observations per group
    alpha: Significance level (default 0.05)
    two_tailed: Whether the test is two-tailed (default True)
    
    Returns:
    Statistical power as a float rounded to 4 decimal places
    """
    ncp = effect_size*math.sqrt(sample_size_per_group/2)
    match two_tailed:
        case True:
            q = NormalDist(mu=0, sigma=1).inv_cdf(1-alpha/2)
            power = 1 - cdf(q - ncp) + cdf(-q - ncp)
        case False:
            q = NormalDist(mu=0, sigma=1).inv_cdf(1-alpha)
            power = 1 - cdf(q - ncp)
    return round(power, 4)