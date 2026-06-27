def prepare_eval_input(tokens: list, mode: str, reserved_output: int) -> list:
    """
    Truncate a token list to fit the context window of the given reasoning mode.
    """
    match mode:
        case "non-think":
            C = 8192
        case "high":
            C = 131072
        case "max":
            C = 393216
        case _:
            raise ValueError
    L = C - reserved_output

    if L<=0:
        return []
    elif len(tokens) > L:
        return tokens[-L:]
    else:
        return tokens