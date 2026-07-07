import numpy as np

def deterministic_hash(s):
    '''Converts a string to a deterministic integer.'''
    h = 0
    for c in str(s):
        h = (h * 31 + ord(c)) % (2**31)
    return h

def create_hv(dim, seed):
    '''Creates a bipolar hypervector of given dimension using the seed.'''
    np.random.seed(seed % (2**32 - 1))
    return np.random.choice([-1, 1], dim)

def create_row_hv(row, dim, random_seeds):
    '''Create composite hypervector for a dataset row.
    
    Hint: For each feature, the value seed should combine the base seed
    with the hashed value using modular arithmetic.
    '''
    bundle = []
    for name, value in row.items():
        h_value = deterministic_hash(value)

        hv_name = create_hv(dim, random_seeds[name])
        hv_value = create_hv(dim, random_seeds[name] + h_value)

        bound = hv_name * hv_value
        bundle.append(bound)
    bundle = np.sum(bundle, axis=0)

    return np.where(bundle>=0, 1, -1)

