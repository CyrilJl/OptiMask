import numpy as np
import pandas as pd
import pytest

from optimask import OptiMask


def generate_random(m, n, ratio):
    """Missing at random arrays"""
    arr = np.zeros((m, n))
    nan_count = int(ratio * m * n)
    indices = np.random.choice(m * n, nan_count, replace=False)
    arr.flat[indices] = np.nan
    return arr


@pytest.fixture
def opti_mask_instance():
    return OptiMask()


def test_solve_with_numpy_array(opti_mask_instance):
    m, n = 350, 50
    ratio = 0.2
    n = 150
    for _ in range(n):
        input_data = generate_random(m, n, ratio)
        rows, cols = opti_mask_instance.solve(input_data)
        assert np.all(np.isfinite(input_data[rows][:, cols]))


def test_solve_with_pandas_dataframe(opti_mask_instance):
    m, n = 350, 50
    ratio = 0.2
    n = 150
    for _ in range(n):
        input_data = pd.DataFrame(generate_random(m, n, ratio))
        rows, cols = opti_mask_instance.solve(input_data)
        assert np.all(np.isfinite(input_data.loc[rows, cols]))


def test_no_nan(opti_mask_instance):
    m, n = 100, 75
    x = np.ones((m, n))
    r, c = opti_mask_instance.solve(x)
    assert np.allclose(r, np.arange(m))
    assert np.allclose(c, np.arange(n))
