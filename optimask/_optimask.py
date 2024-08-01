# -*- coding: utf-8 -*-
"""
author: Cyril Joly
"""

from typing import Tuple, Union

import numpy as np
import pandas as pd
from numba import bool_, njit, prange, uint32
from numba.types import UniTuple

from ._misc import EmptyInputError, InvalidDimensionError, OptiMaskAlgorithmError, check_params, warning

__all__ = ["OptiMask"]


class OptiMask:
    """
    A class to solve the optimal problem of removing NaNs for a 2D array or DataFrame.

    Attributes:
        n_tries (int): Number of tries to find the optimal solution.
        max_steps (int): Maximum number of steps for the optimization process.
        random_state (int, optional): Seed for the random number generator to ensure reproducibility.
        verbose (bool): If True, prints detailed information about the optimization process.

    """

    def __init__(self, n_tries=5, max_steps=16, random_state=None, verbose=False):
        self.n_tries = n_tries
        self.max_steps = max_steps
        self.random_state = random_state
        self.verbose = bool(verbose)

    @staticmethod
    @njit(uint32[:](uint32[:], uint32[:], uint32), boundscheck=False, cache=True)
    def groupby_max(a, b, n):
        size_a = len(a)
        ret = np.zeros(n, dtype=np.uint32)
        for k in range(size_a):
            ak = a[k]
            ret[ak] = max(ret[ak], b[k]+1)
        return ret

    @staticmethod
    @njit(UniTuple(uint32[:], 2)(uint32[:], uint32[:], uint32, uint32), boundscheck=False, cache=True)
    def cross_groupby_max(a, b, m, n):
        size_a = len(a)
        size_b = len(b)
        s = max(size_a, size_b)
        ret_a = np.zeros(m, dtype=np.uint32)
        ret_b = np.zeros(n, dtype=np.uint32)
        for k in range(s):
            ak = a[k]
            bk = b[k]
            if k < size_a:
                ret_a[ak] = max(ret_a[ak], bk+1)
            if k < size_b:
                ret_b[bk] = max(ret_b[bk], ak+1)
        return ret_a, ret_b

    def _verbose(self, msg):
        if self.verbose:
            warning(msg)

    @staticmethod
    @njit(bool_(uint32[:]), boundscheck=False, cache=True)
    def is_decreasing(h):
        for i in range(len(h) - 1):
            if h[i] < h[i + 1]:
                return False
        return True

    @classmethod
    def is_pareto_ordered(cls, hy, hx):
        return cls.is_decreasing(hx) and cls.is_decreasing(hy)

    @staticmethod
    @njit(uint32[:](uint32[:], uint32[:]), parallel=True, boundscheck=False, cache=True)
    def numba_apply_permutation(p, x):
        n = p.size
        m = x.size
        rank = np.empty(n, dtype=np.uint32)
        result = np.empty(m, dtype=np.uint32)

        for i in prange(n):
            rank[p[i]] = i

        for i in prange(m):
            result[i] = rank[x[i]]
        return result

    @staticmethod
    @njit((uint32[:], uint32[:]), parallel=True, boundscheck=False, cache=True)
    def numba_apply_permutation_inplace(p, x):
        n = p.size
        rank = np.empty(n, dtype=np.uint32)

        for i in prange(n):
            rank[p[i]] = i

        for i in prange(x.size):
            x[i] = rank[x[i]]

    @classmethod
    def apply_permutation(cls, p, x, inplace: bool):
        if inplace:
            cls.numba_apply_permutation_inplace(p, x)
        else:
            return cls.numba_apply_permutation(p, x)

    @staticmethod
    @njit(UniTuple(uint32[:], 2)(uint32[:], uint32[:], uint32[:]), parallel=True, boundscheck=False, cache=True)
    def apply_p_step(p_step, a, b):
        ret_a = np.empty(a.size, dtype=np.uint32)
        ret_b = np.empty(b.size, dtype=np.uint32)
        for k in prange(a.size):
            pk = p_step[k]
            ret_a[k] = a[pk]
            ret_b[k] = b[pk]
        return ret_a, ret_b

    @staticmethod
    def _get_largest_rectangle(heights, m, n):
        areas = (m - heights) * (n - np.arange(len(heights)))
        i0 = np.argmax(areas)
        return i0, heights[i0], areas[i0]

    @staticmethod
    @njit(boundscheck=False, cache=True)
    def _preprocess(x):
        m, n = x.shape
        iy, ix = [], []
        cols_index_mapper = -np.ones(n)
        rows_with_nan = np.zeros(m, dtype=np.bool_)
        n_rows_with_nan = 0
        n_cols_with_nan = 0
        for i in range(m):
            for j in range(n):
                if np.isnan(x[i, j]):
                    rows_with_nan[i] = True

                    iy.append(n_rows_with_nan)

                    if cols_index_mapper[j] >= 0:
                        ix.append(cols_index_mapper[j])
                    else:
                        ix.append(n_cols_with_nan)
                        cols_index_mapper[j] = n_cols_with_nan
                        n_cols_with_nan += 1

            if rows_with_nan[i]:
                n_rows_with_nan += 1

        iy, ix = np.array(iy).astype(np.uint32), np.array(ix).astype(np.uint32)
        rows_with_nan = np.flatnonzero(rows_with_nan).astype(np.uint32)
        cols_with_nan = np.flatnonzero(cols_index_mapper >= 0)[cols_index_mapper[cols_index_mapper >= 0].argsort()].astype(np.uint32)
        return iy, ix, rows_with_nan, cols_with_nan

    def _trial(self, rng, m_nan, n_nan, iy, ix, m, n):
        p_rows = rng.permutation(m_nan).astype(np.uint32)
        p_cols = rng.permutation(n_nan).astype(np.uint32)
        iy_trial = self.apply_permutation(p_rows, iy, inplace=False)
        ix_trial = self.apply_permutation(p_cols, ix, inplace=False)

        hy, hx = self.cross_groupby_max(iy_trial, ix_trial, m_nan, n_nan)
        step = 0
        is_pareto_ordered = False
        while not is_pareto_ordered and step < self.max_steps:
            axis = step % 2
            step += 1
            if axis == 0:
                p_step = (-hy).argsort().astype(np.uint32)
                self.apply_permutation(p_step, iy_trial, inplace=True)
                p_rows, hy = self.apply_p_step(p_step, p_rows, hy)
                hx = self.groupby_max(ix_trial, iy_trial, n_nan)
                is_pareto_ordered = self.is_decreasing(hx)
            else:
                p_step = (-hx).argsort().astype(np.uint32)
                self.apply_permutation(p_step, ix_trial, inplace=True)
                hy = self.groupby_max(iy_trial, ix_trial, m_nan)
                p_cols, hx = self.apply_p_step(p_step, p_cols, hx)
                is_pareto_ordered = self.is_decreasing(hy)

        if not self.is_pareto_ordered(hy, hx):
            raise OptiMaskAlgorithmError("An error occurred while calculating optimal permutations. "
                                         "You can try again with a larger `max_steps` value.")
        else:
            i0, j0, area = self._get_largest_rectangle(hx, m, n)
            return area, i0, j0, p_rows, p_cols

    def _solve(self, x):
        m, n = x.shape

        if m == 1:
            return np.arange(m), np.flatnonzero(np.isfinite(x.ravel()))
        if n == 1:
            return np.flatnonzero(np.isfinite(x.ravel())), np.arange(n)

        iy, ix, rows_with_nan, cols_with_nan = self._preprocess(x)
        m_nan, n_nan = len(rows_with_nan), len(cols_with_nan)
        if len(iy) == 0:
            return np.arange(m), np.arange(n)

        if len(iy) == m*n:
            self._verbose('The array is full of NaNs.')
            if m <= n:
                return np.array([]), np.arange(n)
            else:
                return np.arange(m), np.array([])

        if len(rows_with_nan) == 1:
            if n-n_nan <= n_nan*(m-m_nan):
                return np.setdiff1d(np.arange(m), rows_with_nan), np.arange(n)
            else:
                return np.arange(m), np.setdiff1d(np.arange(n), cols_with_nan)

        if len(cols_with_nan) == 1:
            if m-m_nan <= m_nan*(n-n_nan):
                return np.arange(m), np.setdiff1d(np.arange(n), cols_with_nan)
            else:
                return np.setdiff1d(np.arange(m), rows_with_nan), np.arange(n)

        else:
            rng = np.random.default_rng(seed=self.random_state)
            area_max = -1
            for k in range(self.n_tries):
                area, i0, j0, p_rows, p_cols = self._trial(rng, m_nan, n_nan, iy, ix, m, n)
                self._verbose(f"\tTrial {k+1} : submatrix of size {m-j0}x{n-i0} ({area} elements) found.")
                if area > area_max:
                    area_max = area
                    opt = i0, j0, p_rows, p_cols

            i0, j0, p_rows, p_cols = opt
            self._verbose(f"Result: the largest submatrix found is of size {m-j0}x{n-i0} ({area_max} elements) found.")

            rows_to_keep = np.setdiff1d(np.arange(m, dtype=np.uint32), rows_with_nan[p_rows[:j0]])
            cols_to_keep = np.setdiff1d(np.arange(n, dtype=np.uint32), cols_with_nan[p_cols[:i0]])
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
            raise InvalidDimensionError("For a numpy array, 'X' must have ndim == 2.")

        if X.size == 0:
            raise EmptyInputError("`X` is empty.")

        rows, cols = self._solve(np.asarray(X))

        if np.isnan(np.asarray(X)[np.ix_(rows, cols)]).any():
            raise OptiMaskAlgorithmError("The OptiMask algorithm encountered an error.")

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
