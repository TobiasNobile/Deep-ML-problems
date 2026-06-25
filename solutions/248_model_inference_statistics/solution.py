import math

def calculate_inference_stats(latencies_ms: list) -> dict:
    """
    Calculate inference statistics for model monitoring.
    
    Args:
        latencies_ms: list of latency measurements in milliseconds
    
    Returns:
        dict with keys: 'throughput_per_sec', 'avg_latency_ms', 'p50_ms', 'p95_ms', 'p99_ms'
        All values rounded to 2 decimal places.
    """
    if not latencies_ms:
        return {}
    latencies_ms.sort()
    n = len(latencies_ms)
    avg_latency_ms = sum(latencies_ms)/n
    throughput = 1000/avg_latency_ms

    index_p50 = 50/100 * (n - 1)
    p50 = latencies_ms[math.floor(index_p50)] + (index_p50 - math.floor(index_p50))*(latencies_ms[math.ceil(index_p50)] - latencies_ms[math.floor(index_p50)])

    index_p95 = 95/100 * (n - 1)
    p95 = latencies_ms[math.floor(index_p95)] + (index_p95 - math.floor(index_p95))*(latencies_ms[math.ceil(index_p95)] - latencies_ms[math.floor(index_p95)])

    index_p99 = 99/100 * (n - 1)
    p99 = latencies_ms[math.floor(index_p99)] + (index_p99 - math.floor(index_p99))*(latencies_ms[math.ceil(index_p99)] - latencies_ms[math.floor(index_p99)])

    return {
        'throughput_per_sec': round(throughput, 2), 
        'avg_latency_ms': round(avg_latency_ms, 2), 
        'p50_ms': round(p50, 2), 
        'p95_ms': round(p95, 2), 
        'p99_ms': round(p99, 2)
    }
