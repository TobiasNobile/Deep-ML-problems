import numpy as np

def zero_copy_env_simulation(num_envs: int, obs_dim: int, step_data: list, commands: list) -> dict:
    """
    Simulate a vectorized C environment with zero-copy buffer management.

    Args:
        num_envs: Number of parallel environments.
        obs_dim: Dimension of each observation vector.
        step_data: List of (obs_2d, rewards, dones) tuples for each timestep.
        commands: List of command tuples to execute.

    Returns:
        Dictionary with keys: 'reads', 'alias_checks', 'total_copies',
        'total_views', 'total_bytes_saved', 'buffer_reallocs'.
    """
    reads, alias_checks = [], []
    total_copies, total_views = 0, 0
    total_bytes_saved, buffer_reallocs = 0, 0

    snapshots = {}
    buffer_ops = np.zeros((num_envs, obs_dim), dtype = np.float64)
    buffer_rew = np.zeros((num_envs,), dtype = np.float64)
    buffer_don = np.zeros((num_envs,), dtype = np.int8)
    i_data = 0
    for i in range(len(commands)):
        command_tuple = commands[i]
        command = command_tuple[0]

        if command == "store_view":
            snapshots[command_tuple[1]] = buffer_ops
            total_views += 1
            total_bytes_saved += num_envs*obs_dim*8
        elif command == "store_copy":
            snapshots[command_tuple[1]] = buffer_ops.copy()
            total_copies += 1
        elif command == "read":
            reads.append(np.round(snapshots[command_tuple[1]], 4).tolist())
        elif command == "read_buffer":
            reads.append(np.round(buffer_ops, 4).tolist())
        elif command == "step":
            buffer_ops[:] = step_data[i_data][0]
            buffer_rew[:] = step_data[i_data][1]
            buffer_don[:] = step_data[i_data][2]
            i_data += 1
        elif command == "check_alias":
            memory_shared = np.shares_memory(snapshots[command_tuple[1]], buffer_ops)
            alias_checks.append(memory_shared)
        elif command == "auto_reset":
            mask = buffer_don > 0
            buffer_ops[mask] = 0
            buffer_rew[mask] = 0

    return {
        "reads": reads, 
        "alias_checks": alias_checks, 
        "total_copies": total_copies, 
        "total_views": total_views, 
        "total_bytes_saved": total_bytes_saved, 
        "buffer_reallocs": buffer_reallocs
        }
    