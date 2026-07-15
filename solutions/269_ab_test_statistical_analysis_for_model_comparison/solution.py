import numpy as np
from math import *

def analyze_ab_test(
    control_outcomes: list, 
    treatment_outcomes: list, 
    confidence_level: float = 0.95, min_detectable_effect: float = 0.02) -> dict:
    """
    Analyze A/B test results for model comparison with statistical rigor.
    
    Args:
        control_outcomes: List of binary outcomes (0 or 1) for control group
        treatment_outcomes: List of binary outcomes (0 or 1) for treatment group
        confidence_level: Confidence level for statistical tests (default 0.95)
        min_detectable_effect: Minimum absolute effect size considered practically significant
    
    Returns:
        dict with statistical analysis results and recommendation
    """
    if not (control_outcomes and treatment_outcomes):
        return {}
    control, treatment = np.array(control_outcomes), np.array(treatment_outcomes)
    results = {}

    results["control_rate"] = np.sum(control)/len(control)
    results["treatment_rate"] = np.sum(treatment)/len(treatment)

    results["absolute_lift"] = np.sum(treatment - control)/len(treatment)

    p_pool = len(control)*results["control_rate"]
    p_pool += len(treatment)*results["treatment_rate"]
    p_pool /= len(control) + len(treatment)

    results["z_statistic"] = results["treatment_rate"] - results["control_rate"]
    results["z_statistic"] /= np.sqrt(p_pool*(1-p_pool)*(1/len(control) + 1/len(treatment)))

    cdf = 0.5  * (1+erf(abs(results["z_statistic"])/sqrt(2)))
    results["p_value"] = 2*(1 - cdf)

    results["statistically_significant"] = results["p_value"] < 1 - confidence_level

    sample_size_min = 2*(results["z_statistic"]/min_detectable_effect)**2 *  results["treatment_rate"]* (1 - results["treatment_rate"])
    results["practically_significant"] = sample_size_min>=sample_size_min

    results["required_sample_size"] = sample_size_min
    
    if results["statistically_significant"] and results["practically_significant"] and results["absolute_lift"] > 0:
        results["recommendation"] = "launch_treatment"
    elif results["statistically_significant"] and (results["absolute_lift"] <= 0 or not results["practically_significant"]):
        results["recommendation"] = "keep_control"
    elif not results["statistically_significant"] or not results["practically_significant"]:
        results["recommendation"] = "continue_testing"
    
    results['control_rate'] = float(results['control_rate'])
    results['treatment_rate'] = float(results['treatment_rate'])
    results['z_statistic'] = round(float(results['z_statistic']), 4)
    results['practically_significant'] = bool(results['z_statistic'])

    return results