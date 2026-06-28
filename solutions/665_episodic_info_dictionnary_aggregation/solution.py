import numpy as np

def aggregate_episodic_info(infos: list) -> dict:
    """
    Aggregate episodic statistics from a list of step-level info dictionaries.
    
    Args:
        infos: List of info dictionaries from environment steps.
               Each dict may contain an 'episode' key with sub-dict
               having 'r' (total reward) and 'l' (length) keys.
    
    Returns:
        Dictionary with aggregated episode statistics.
    """
    data = []
    for info in infos:
        episode = info.get("episode", None)
        if episode:
            r, l = episode["r"], episode["l"]
            data.append((r, l))
    
    return {
        "num_episodes": len(data) if data else 0,
        "mean_reward": np.mean(data, axis=0)[0] if data else 0,
        "mean_length": np.mean(data, axis=0)[1] if data else 0,
        "min_reward": np.min(data, axis=0)[0] if data else 0,
        "max_reward": np.max(data, axis=0)[0] if data else 0,
        "min_length": np.min(data, axis=0)[1] if data else 0,
        "max_length": np.max(data, axis=0)[1] if data else 0,
    }