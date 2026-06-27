def compute_arithmetic_intensity(flops: float, bytes_accessed: float, peak_performance: float, peak_bandwidth: float) -> dict:
    """
    Analyze a computational kernel using the Roofline Model.
    
    Args:
        flops: Total floating-point operations of the kernel
        bytes_accessed: Total bytes transferred to/from memory
        peak_performance: Hardware peak compute throughput (FLOP/s)
        peak_bandwidth: Hardware peak memory bandwidth (bytes/s)
    
    Returns:
        Dictionary with arithmetic_intensity, ridge_point, bottleneck,
        achieved_performance, and utilization_percent
    """
    kernel_analysis = {}
    kernel_analysis["arithmetic_intensity"] = flops / bytes_accessed
    kernel_analysis["ridge_point"] = peak_performance / peak_bandwidth
    if kernel_analysis["arithmetic_intensity"] < kernel_analysis["ridge_point"]:
        kernel_analysis["bottleneck"] = "memory-bound"
    else:
        kernel_analysis["bottleneck"] = "compute-bound"

    kernel_analysis["achieved_performance"] = min(kernel_analysis["arithmetic_intensity"] * peak_bandwidth, peak_performance)
    kernel_analysis["utilization_percent"] = kernel_analysis["achieved_performance"] /peak_performance*100
    return kernel_analysis

