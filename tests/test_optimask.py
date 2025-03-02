import numpy as np
import pandas as pd
import polars as pl
import pytest

from optimask import OptiMask


def generate_random(m, n, ratio):
    """Missing at random arrays"""
    arr = np.zeros((m, n), dtype=np.float32)
    nan_count = int(ratio * m * n)
    indices = np.random.choice(m * n, nan_count, replace=False)
    arr.flat[indices] = np.nan
    return arr


@pytest.fixture
def opti_mask_instance() -> OptiMask:
    return OptiMask(verbose=1)


def test_solve_with_numpy_array(opti_mask_instance):
    m, n = 350, 50
    ratio = 0.2
    n_tries = 50
    for _ in range(n_tries):
        input_data = generate_random(m, n, ratio)
        rows, cols = opti_mask_instance.solve(input_data, check_result=True)
        assert np.all(np.isfinite(input_data[rows][:, cols]))
        sub_data = opti_mask_instance.solve(input_data, check_result=True, return_data=True)
        assert np.all(np.isfinite(sub_data))
        assert isinstance(sub_data, np.ndarray)


def test_solve_with_pandas_dataframe(opti_mask_instance):
    m, n = 350, 50
    ratio = 0.2
    n_tries = 50
    for _ in range(n_tries):
        input_data = pd.DataFrame(generate_random(m, n, ratio))
        rows, cols = opti_mask_instance.solve(input_data, check_result=True)
        assert np.all(np.isfinite(input_data.loc[rows, cols]))
        sub_data = opti_mask_instance.solve(input_data, check_result=True, return_data=True)
        assert np.all(np.isfinite(sub_data))
        assert isinstance(sub_data, pd.DataFrame)


def test_solve_with_polars_dataframe(opti_mask_instance):
    m, n = 350, 50
    ratio = 0.2
    n_tries = 50
    for _ in range(n_tries):
        input_data = pl.DataFrame(generate_random(m, n, ratio))
        rows, cols = opti_mask_instance.solve(input_data, check_result=True)
        assert np.all(np.isfinite(input_data[rows][:, cols]))
        sub_data = opti_mask_instance.solve(input_data, check_result=True, return_data=True)
        assert np.all(np.isfinite(sub_data))
        assert isinstance(sub_data, pl.DataFrame)


def test_no_nan(opti_mask_instance):
    m, n = 100, 75
    x = np.ones((m, n))
    r, c = opti_mask_instance.solve(x, check_result=True)
    assert np.allclose(r, np.arange(m))
    assert np.allclose(c, np.arange(n))

    assert np.allclose(x, opti_mask_instance.solve(x, return_data=True))


def test_full_nans(opti_mask_instance):
    x = np.full(shape=(500, 1000), fill_value=np.nan)
    r, c = opti_mask_instance.solve(x)
    assert len(r) == 0
    assert len(c) == 1000


def test_one_col(opti_mask_instance):
    m = 100
    x = np.arange(m, dtype=float)[:, None]
    x[-1, 0] = np.nan

    assert np.allclose(np.arange(m - 1)[:, None], opti_mask_instance.solve(x, return_data=True, check_result=True))


def test_one_row(opti_mask_instance):
    m = 100
    x = np.arange(m, dtype=float)[None, :]
    x[0, -1] = np.nan

    assert np.allclose(np.arange(m - 1)[None, :], opti_mask_instance.solve(x, return_data=True, check_result=True))


def test_nan_in_one_col(opti_mask_instance):
    for m, n in ((75, 2), (75, 500)):
        x = np.zeros(shape=(m, n))
        x[:25, 0] = np.nan
        opti_mask_instance.solve(X=x, check_result=True)


def test_nan_in_one_row(opti_mask_instance):
    for m, n in ((500, 75), (2, 75)):
        x = np.zeros(shape=(m, n))
        x[0, :25] = np.nan
        opti_mask_instance.solve(X=x, check_result=True)


def test_seed():
    """The test between results of two different seeds can be subject to
    collisions, but it is unlikely.
    """
    X = generate_random(m=10_000, n=500, ratio=0.025)
    om1 = OptiMask(random_state=99)
    om2 = OptiMask(random_state=99)
    om3 = OptiMask(random_state=999)
    rows1, cols1 = om1.solve(X, check_result=True)
    rows2, cols2 = om2.solve(X, check_result=True)
    rows3, cols3 = om3.solve(X, check_result=True)
    assert np.allclose(rows1, rows2)
    assert np.allclose(cols1, cols2)
    assert (len(rows1) != len(rows3)) or (not np.allclose(rows1, rows3))
    assert (len(cols1) != len(cols3)) or (not np.allclose(cols1, cols3))
