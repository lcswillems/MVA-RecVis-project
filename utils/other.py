import numpy as np

def va(action):
    """Vectorize the dict action."""

    return np.concatenate((action['linear_velocity'], action['grip_velocity']))

def aaT(a):
    """Take a numpy vector a and compute the matrix a^T a."""

    a = a.reshape(-1, 1)
    return a.dot(a.T)