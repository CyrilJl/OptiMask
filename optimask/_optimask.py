# -*- coding: utf-8 -*-
"""
author: Cyril Joly
"""

from typing import Tuple, Union

import numpy as np
import pandas as pd

from ._misc import check_params, warning
from .optimask_cython import groupby_max, is_decreasing, permutation_index

__all__ = ["OptiMask"]


class OptiMask:
    """
    OptiMask is a class for calculating the optimal rows and columns to retain in a 2D array or DataFrame
    to remove NaN values and preserve the maximum number of non-NaN cells.
    The class uses a heuristic optimization approach, and increasing the value of `n_tries` generally leads to better results,
    potentially reaching or closely approaching the optimal quantity.

    Parameters:
        n_tries (int): The number of optimization attempts. Higher values may lead to better results.
        max_steps (int): The maximum number of steps to perform in each optimization attempt.
        random_state (Union[int, None]): Seed for the random number generator.
        verbose (bool): If True, print verbose information during optimization.

    .. code-block:: python

        from optimask import OptiMask
        import numpy as np

        # Create a matrix with NaN values
        m = 120
        n = 7
        data = np.zeros(shape=(m, n))
        data[24:72, 3] = np.nan
        data[95, :5] = np.nan

        # Solve for the largest submatrix without NaN values
        rows, cols = OptiMask().solve(data)

        # Calculate the ratio of non-NaN values in the result
        coverage_ratio = len(rows) * len(cols) / data.size

        # Check if there are any NaN values in the selected submatrix
        has_nan_values = np.isnan(data[rows][:, cols]).any()

        # Print or display the results
        print(f"Coverage Ratio: {coverage_ratio:.2f}, Has NaN Values: {has_nan_values}")
        # Output: Coverage Ratio: 0.85, Has NaN Values: False
    """

    def __init__(self, n_tries=5, max_steps=32, random_state=None, verbose=False):
        self.n_tries = n_tries
        self.max_steps = max_steps
        self.random_state = random_state
        self.verbose = bool(verbose)

    def _verbose(self, msg):
        if self.verbose:
            warning(msg)

    @staticmethod
    def _sort_by_na_max_index(height: np.ndarray) -> np.ndarray:
        return np.argsort(-height, kind='mergesort').astype(np.int32)

    @staticmethod
    def _get_largest_rectangle(heights, m, n):
        areas = (m - heights) * (n - np.arange(len(heights)))
        i0 = np.argmax(areas)
        return i0, heights[i0], areas[i0]

    @classmethod
    def _is_pareto_ordered(cls, hx, hy):
        return is_decreasing(hx) and is_decreasing(hy)

    def _trial(self, p_rows, p_cols, iy, ix, m, n):
        iy_trial = permutation_index(p_rows)[iy]
        ix_trial = permutation_index(p_cols)[ix]

        step = 0
        h0, h1 = groupby_max(iy_trial, ix_trial), groupby_max(ix_trial, iy_trial)
        while (not self._is_pareto_ordered(h0, h1)) and (step < self.max_steps):
            axis = (step % 2)
            step += 1
            if axis == 0:
                p_step = self._sort_by_na_max_index(h0)
                iy_trial = permutation_index(p_step)[iy_trial]
                p_rows = p_rows[p_step]
                h0, h1 = h0[p_step], groupby_max(ix_trial, iy_trial)
            if axis == 1:
                p_step = self._sort_by_na_max_index(h1)
                ix_trial = permutation_index(p_step)[ix_trial]
                p_cols = p_cols[p_step]
                h0, h1 = groupby_max(iy_trial, ix_trial), h1[p_step]

        if not self._is_pareto_ordered(h0, h1):
            raise ValueError("An error occurred while calculating optimal permutations. "
                             "You can try again with a larger `max_steps` value.")
        else:
            i0, j0, area = self._get_largest_rectangle(h1, m, n)
            return area, i0, j0, p_rows, p_cols

    def _solve(self, x):
        m, n = x.shape
        iy, ix = np.isnan(x).nonzero()
        if len(iy) == 0:
            return np.arange(m), np.arange(n)
        else:
            rng = np.random.default_rng(seed=self.random_state)
            rows_with_nan, iy = np.unique(iy, return_inverse=True)
            cols_with_nan, ix = np.unique(ix, return_inverse=True)
            iy, ix = iy.astype(np.int32), ix.astype(np.int32)
            m_nan, n_nan = len(rows_with_nan), len(cols_with_nan)

            area_max = -1
            for k in range(self.n_tries):
                p_rows = rng.permutation(m_nan)
                p_cols = rng.permutation(n_nan)
                area, i0, j0, p_rows, p_cols = self._trial(p_rows, p_cols, iy, ix, m, n)
                self._verbose(f"\tTrial {k+1} : submatrix of size {m-j0}x{n-i0} ({area} elements) found.")
                if area > area_max:
                    area_max = area
                    opt = i0, j0, p_rows, p_cols

            i0, j0, p_rows, p_cols = opt
            self._verbose(f"Result: the largest submatrix found is of size {m-j0}x{n-i0} ({area_max} elements) found.")

            rows_to_keep = np.setdiff1d(np.arange(m), rows_with_nan[p_rows[:j0]])
            cols_to_keep = np.setdiff1d(np.arange(n), cols_with_nan[p_cols[:i0]])
            return rows_to_keep, cols_to_keep

    def solve(self, X: Union[np.ndarray, pd.DataFrame], return_data: bool = False) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[pd.Index, pd.Index]]:
        """
        Solves the optimal problem of removing NaNs for a 2D array or DataFrame.

        Args:
            X (Union[np.ndarray, pd.DataFrame]): The input 2D array or DataFrame with NaN values.
            return_data (bool): If True, returns the resulting data; otherwise, returns the indices.

        Returns:
            Union[Tuple[np.ndarray, np.ndarray], Tuple[pd.Index, pd.Index]]: If return_data is True, returns the resulting 2D array or DataFrame; otherwise, returns the indices of rows and columns to retain.

        Raises:
            ValueError: If the input data is not a numpy array or a pandas DataFrame, or if the input numpy array does not have ndim==2, or if the OptiMask algorithm encounters an error during optimization.
        """
        check_params(X, types=(np.ndarray, pd.DataFrame))

        if isinstance(X, np.ndarray) and X.ndim != 2:
            raise ValueError("For a numpy array, 'X' must have ndim == 2.")

        rows, cols = self._solve(np.asarray(X))

        if np.isnan(np.asarray(X)[np.ix_(rows, cols)]).any():
            raise ValueError("The OptiMask algorithm encountered an error.")

        if isinstance(X, pd.DataFrame):
            if return_data:
                return X.iloc[rows, cols].copy()
            else:
                if not X.index.is_unique:
                    raise ValueError("The index contains non-unique entries!")
                return X.index[rows].copy(), X.columns[cols].copy()

        if isinstance(X, np.ndarray):
            if return_data:
                return X[rows][:, cols].copy()
            else:
                return rows, cols
