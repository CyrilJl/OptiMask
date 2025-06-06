from time import perf_counter

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
    return OptiMask()


def get_rows_cols(data, rows, cols):
    """Helper function to get sub-data based on data type."""
    if isinstance(data, np.ndarray):
        return data[rows][:, cols]
    elif isinstance(data, pd.DataFrame):
        return data.loc[rows, cols]
    elif isinstance(data, pl.DataFrame):
        return data[rows][:, cols]
    else:
        raise TypeError(f"Unsupported data type: {type(data)}")


def assert_finite_and_type(sub_data, expected_type):
    """Helper function to assert finite values and type."""
    assert np.all(np.isfinite(sub_data))
    assert isinstance(sub_data, expected_type)


@pytest.mark.parametrize("data_type", [np.ndarray, pd.DataFrame, pl.DataFrame])
def test_solve_generic(opti_mask_instance, data_type):
    m, n = 1000, 50
    ratio = 0.1
    n_tries = 15

    for _ in range(n_tries):
        input_data = generate_random(m, n, ratio)
        if data_type is pd.DataFrame:
            input_data = pd.DataFrame(input_data)
        elif data_type is pl.DataFrame:
            input_data = pl.DataFrame(input_data)

        rows, cols = opti_mask_instance.solve(input_data, check_result=True)
        sub_data_extracted = get_rows_cols(input_data, rows, cols)
        assert np.all(np.isfinite(sub_data_extracted))

        sub_data = opti_mask_instance.solve(input_data, check_result=True, return_data=True)
        assert_finite_and_type(sub_data, data_type)


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
    rows1, cols1 = om1.solve(X, check_result=True)
    rows2, cols2 = om2.solve(X, check_result=True)
    assert np.allclose(rows1, rows2)
    assert np.allclose(cols1, cols2)


def test_speed(opti_mask_instance):
    x = generate_random(m=100_000, n=1_000, ratio=0.02)
    print()
    for _ in range(5):
        start = perf_counter()
        opti_mask_instance.solve(X=x)
        print(f"{1e3 * (perf_counter() - start):.2f}ms")


def test_large_arrays(opti_mask_instance):
    x = generate_random(m=100_000, n=1_000, ratio=0.02)
    opti_mask_instance.solve(x)

    x = generate_random(m=1_000, n=100_000, ratio=0.02)
    opti_mask_instance.solve(x)
