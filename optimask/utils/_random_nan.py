import numpy as np


def generate_mar(m, n, ratio) -> np.ndarray:
    """Missing at random 2D array"""
    assert 0 <= ratio <= 1, "Ratio must be a float between 0 and 1."
    return np.random.choice(a=[0, np.nan], size=(m, n), p=[1-ratio, ratio])
