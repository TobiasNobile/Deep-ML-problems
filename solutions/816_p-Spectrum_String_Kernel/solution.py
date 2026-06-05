def string_kernel(s: str, t: str, p: int) -> int:
    """
    Compute the p-spectrum string kernel between two strings.
    """
    hash_s, hash_t = {}, {} 

    for i in range(max(len(s), len(t))-1):
        sub_s, sub_t = s[i:p+i], t[i:p+i]
        hash_s[sub_s] = hash_s.get(sub_s, 0) + 1
        hash_t[sub_t] = hash_t.get(sub_t, 0) + 1

    common_substr = list(set(hash_s).intersection(set(hash_t)))
    K = 0

    return sum(hash_s[sub]*hash_t[sub] for sub in common_substr)