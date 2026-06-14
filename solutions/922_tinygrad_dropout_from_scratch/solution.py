from tinygrad import Tensor

def dropout(x, p, training):
    # TODO: inverted dropout
    if not training or p == 0:
        return x
    m = Tensor.rand(x.shape) >= p
    m = m.cast(x.dtype)
    return m/(1-p)*x