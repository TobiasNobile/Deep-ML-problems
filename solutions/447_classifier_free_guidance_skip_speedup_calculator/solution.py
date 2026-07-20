import numpy as np

def cfg_skip_speedup(
    T: int,
    skip_mask: list[bool],
    time_per_pass: float,
    cond_preds: list[float],
    uncond_preds: list[float],
    guidance_scale: float
) -> dict:
    """
    Analyze the speedup and quality impact of skipping unconditional
    passes in Classifier-Free Guidance diffusion inference.
    
    Args:
        T: Total number of denoising timesteps
        skip_mask: Boolean list; True means skip the unconditional pass at that step
        time_per_pass: Time in milliseconds for a single forward pass
        cond_preds: Conditional model predictions at each timestep
        uncond_preds: Unconditional model predictions at each timestep
        guidance_scale: CFG guidance scale (w)
    
    Returns:
        Dictionary with speedup metrics and guided outputs
    """
    result = {"total_passes_standard": 2*T}

    result["total_passes_skipped"] = skip_mask.count(False)*2 + np.sum(skip_mask)

    result["speedup_ratio"] = result["total_passes_standard"] / result["total_passes_skipped"]

    result["time_saved_ms"] = (result["total_passes_standard"] - result["total_passes_skipped"])*time_per_pass

    outputs_std, outputs_skp = [], []
    max_deviation = 0.0
    unc_cache = 0.0
    for t in range(T):
        o_std = (1 - guidance_scale)*uncond_preds[t] + guidance_scale*cond_preds[t]
        outputs_std.append(o_std)

        if not skip_mask[t]:
             outputs_skp.append(o_std)
             unc_cache = uncond_preds[t]
        else:
            o_skp = (1 - guidance_scale)*unc_cache + guidance_scale*cond_preds[t]
            outputs_skp.append(o_skp)
            max_deviation = max(max_deviation, abs(o_std - o_skp))
    
    result["guided_outputs_standard"] = outputs_std
    result["guided_outputs_skipped"] = outputs_skp
    result["max_output_deviation"] = round(max_deviation,4)

    return result