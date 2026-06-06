import numpy as np

def verify_code_execution(
    test_cases: list[dict],
    numeric_tolerance: float = 1e-6
) -> dict:
    """
    Verify code execution results for a programming benchmark.
    
    Args:
        test_cases: List of dicts with keys:
            - 'expected': Expected output string
            - 'actual': Actual output string (or None if execution failed)
            - 'status': 'success', 'error', or 'timeout'
        numeric_tolerance: Tolerance for floating-point comparisons
        
    Returns:
        Dict with keys:
            - 'pass_rate': Proportion of passed tests (float, rounded to 4 decimals)
            - 'error_rate': Proportion of execution errors (float, rounded to 4 decimals)
            - 'passed_count': Number of passed tests (int)
            - 'total_count': Total number of tests (int)
            - 'verdicts': List of 'pass', 'fail', or 'error' for each test
    """
    verdicts = []
    if test_cases:
        for result in test_cases:
            if result["status"] == "error" or result["actual"] == None:
                verdicts.append("error")
            else:
                try:
                    difference = abs(float(result["expected"]) - float(result["actual"]))
                    if difference <= numeric_tolerance:
                        verdicts.append("pass")
                    else:
                        verdicts.append("fail")
                except ValueError:
                    if result["expected"].strip() == result["actual"].strip():
                        verdicts.append("pass")
                    else:
                        verdicts.append("fail")
                
        result = {"pass_rate": round(verdicts.count("pass")/len(verdicts), 4),
                "error_rate": round(verdicts.count("error")/len(verdicts), 4),
                "passed_count": verdicts.count("pass"),
                "total_count": len(verdicts),
                "verdicts": verdicts}
    else:
        result = {"pass_rate": 0.0,
                "error_rate": 0.0,
                "passed_count": 0.0,
                "total_count": 0,
                "verdicts": verdicts}
    
    return result
        