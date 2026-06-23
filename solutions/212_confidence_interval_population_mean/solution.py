import numpy as np

# numpy has no t-distribution quantile function and scipy is not available.
# Use this lookup table to get t-critical values for the (df, p) pairs
# needed by the test cases, where p = 1 - alpha/2.
T_TABLE = {
    (4, 0.95):  2.1318467863998393,
    (4, 0.975): 2.7764451051977987,
    (4, 0.995): 4.604094871415387,
    (5, 0.95):  2.0150483726691575,
    (5, 0.975): 2.5705818366147395,
    (5, 0.995): 4.032142983557536,
    (6, 0.95):  1.9431802803927816,
    (6, 0.975): 2.4469118511449624,
    (6, 0.995): 3.707428021324907,
    (7, 0.95):  1.8945786050613064,
    (7, 0.975): 2.3646242510102993,
    (7, 0.995): 3.4994832973505026,
}


def confidence_interval(data: list[float], confidence_level: float = 0.95) -> dict:
    """
    Calculate confidence interval for population mean.

    Args:
        data: Sample data
        confidence_level: Confidence level (default 0.95)

    Returns:
        Dictionary containing:
        - mean: Sample mean (point estimate)
        - standard_error: Standard error of the mean
        - margin_of_error: Margin of error
        - lower_bound: Lower bound of CI
        - upper_bound: Upper bound of CI
        - confidence_level: Confidence level used
    """
    data = np.array(data)
    n = len(data)
    x_mean = np.mean(data)

    se = np.sqrt((np.var(data)*n/(n-1)))/np.sqrt(n)

    alpha = 1 - confidence_level
    df = n - 1
    t = T_TABLE[(df, 1-alpha/2)]

    margin_error = t*se
    lb, ub = x_mean - margin_error, x_mean + margin_error

    return {
        'mean': x_mean,
        'standard_error': se, 
        'margin_of_error': margin_error, 
        'lower_bound': lb, 
        'upper_bound': ub, 
        'confidence_level': confidence_level
    }

