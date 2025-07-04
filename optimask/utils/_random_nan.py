import numpy as np


def generate_mar(m, n, ratio, rng=None) -> np.ndarray:
    """Missing at random 2D array, filled with 0s and NaNs"""
    rng = np.random.default_rng(rng)
    if ratio < 0 or 1 < ratio:
        raise ValueError("Ratio must be a float between 0 and 1.")
    return rng.choice(a=[0, np.nan], size=(m, n), p=[1 - ratio, ratio])
