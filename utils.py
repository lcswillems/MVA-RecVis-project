def aaT(a):
    """Take a numpy vector a and compute the matrix a^T a."""

    a = a.reshape(-1, 1)
    return a.dot(a.T)