import numpy as np
import math

# scipy is not available in this environment. The two helpers below compute
# the regularized incomplete beta function via the Numerical Recipes
# continued-fraction algorithm — you'll need them to compute the p-value
# from the t-distribution.
def _betacf(a, b, x):
    """Continued fraction for the incomplete beta function."""
    max_iter, eps = 200, 3e-12
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < 1e-30: d = 1e-30
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30: d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30: c = 1e-30
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30: d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30: c = 1e-30
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps: break
    return h

def _betainc(a, b, x):
    """Regularized incomplete beta function I_x(a, b)."""
    if x <= 0.0: return 0.0
    if x >= 1.0: return 1.0
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(math.log(x) * a + math.log(1.0 - x) * b - lbeta)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def two_sample_t_test(sample1: list[float], sample2: list[float],
                      alpha: float = 0.05) -> dict:
    """
    Perform a two-sample independent t-test (Welch's t-test).

    Args:
        sample1: First sample data
        sample2: Second sample data
        alpha: Significance level (default 0.05)

    Returns:
        Dictionary containing:
        - t_statistic: The calculated t-statistic
        - p_value: Two-tailed p-value
        - degrees_of_freedom: Degrees of freedom (Welch-Satterthwaite)
        - reject_null: Boolean, whether to reject null hypothesis
        - cohens_d: Effect size (Cohen's d)
    """
    sample1, sample2 = np.array(sample1), np.array(sample2)
    n1, n2 = len(sample1), len(sample2)
    mean_1, mean_2 = np.mean(sample1), np.mean(sample2)
    var_1, var_2 = np.var(sample1)*n1/(n1-1), np.var(sample2)*n2/(n2-1)

    std_error = np.sqrt(var_1/n1 + var_2/n2)
    t = (mean_1 - mean_2)/std_error

    df = std_error**4
    df /= (((var_1/n1)**2)/(n1-1) + ((var_2/n2)**2)/(n2-1))

    a, b = df / 2, df / 2
    x = 1/2 + 1/2*abs(t)/np.sqrt(df + t**2)
    p_value = 2*(1-_betainc(a, b, x))

    reject_null = True if p_value < alpha else False

    s_pooled = np.sqrt(((n1-1)*var_1 + (n2-1)*var_2) / (n1 + n2 - 2))
    cohens_d = (mean_1 - mean_2)/s_pooled

    return {
        't_statistic': t,
        'p_value': p_value,
        'degrees_of_freedom': df,
        'reject_null': reject_null,
        'cohens_d': cohens_d
    }
