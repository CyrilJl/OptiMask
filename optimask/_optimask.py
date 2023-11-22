# -*- coding: utf-8 -*-

# author: Cyril Joly

from typing import Tuple, Union

import numpy as np
import pandas as pd

__all__ = ["OptiMask"]


class OptiMask:
    """
    OptiMask is a class for calculating the optimal rows and columns to retain in a 2D array or DataFrame
    to remove NaN values and preserve the maximum number of non-NaN cells.
    The class uses a heuristic optimization approach, and increasing the value of `n_tries` generally leads to better results,
    potentially reaching or closely approaching the optimal quantity.

    Parameters:
    - n_tries (int): The number of optimization attempts. Higher values may lead to better results.
    - max_steps (int): The maximum number of steps to perform in each optimization attempt.
    - random_state (Union[int, None]): Seed for the random number generator.

    Attributes:
    - n_tries (int): The number of optimization attempts.
    - max_steps (int): The maximum number of steps to perform in each optimization attempt.
    - random_state (Union[int, None]): Seed for the random number generator.
    """

    def __init__(self, n_tries=10, max_steps=32, random_state=None, verbose=False):
        self.n_tries = n_tries
        self.max_steps = max_steps
        self.random_state = random_state
        self.verbose = bool(verbose)

    def _verbose(self, msg):
        if self.verbose:
            print(msg)

    @staticmethod
    def _find_nan_indices(x):
        nan_rows, nan_cols = np.nonzero(x)
        nan_rows = np.unique(nan_rows)
        nan_cols = np.unique(nan_cols)
        return x[nan_rows][:, nan_cols], nan_rows, nan_cols

    @classmethod
    def _heights(cls, x, axis=0):
        return np.argmax(np.cumsum(x, axis=axis), axis=axis) + 1

    @staticmethod
    def _is_decreasing(x) -> bool:
        return all(x[:-1] >= x[1:])

    @classmethod
    def _sort_by_na_max_index(cls, height) -> np.ndarray:
        return np.argsort(-height, kind='mergesort')

    @classmethod
    def _get_largest_rectangle(cls, heights, m, n):
        areas = (m - heights) * (n - np.arange(len(heights)))
        i0 = np.argmax(areas)
        return heights[i0], i0, areas[i0]

    @classmethod
    def _is_pareto_ordered(cls, hx, hy):
        return cls._is_decreasing(hx) and cls._is_decreasing(hy)

    @classmethod
    def _process_step(cls, x, height, axis):
        p = cls._sort_by_na_max_index(height)
        if axis == 0:
            return x[:, p], p
        if axis == 1:
            return x[p], p

    @staticmethod
    def _random_start(xp, rng):
        p_rows = rng.permutation(xp.shape[0])
        p_cols = rng.permutation(xp.shape[1])
        return xp[p_rows][:, p_cols], p_rows, p_cols

    def _compute_permutations(self, xpp, p_rows, p_cols):
        step = 0
        heights = self._heights(xpp, axis=0), self._heights(xpp, axis=1)
        while (not self._is_pareto_ordered(*heights)) and (step < self.max_steps):
            axis = (step % 2)
            step += 1
            xpp, p_step = self._process_step(xpp, heights[axis], axis=axis)
            if axis == 0:
                p_cols = p_cols[p_step]
                heights = (heights[0][p_step], self._heights(xpp, axis=1))
            if axis == 1:
                p_rows = p_rows[p_step]
                heights = (self._heights(xpp, axis=0), heights[1][p_step])

        if not self._is_pareto_ordered(*heights):
            raise ValueError("An error occurred while calculating optimal permutations. "
                             "You can try again with a larger `max_steps` value.")
        else:
            return xpp, p_rows, p_cols, heights

    def _trial(self, xp, m, n, rng):
        xpp, p_rows, p_cols = self._random_start(xp=xp, rng=rng)
        xpp, p_rows, p_cols, heights = self._compute_permutations(xpp, p_rows, p_cols)
        i0, j0, area = self._get_largest_rectangle(heights[0], m, n)
        return area, i0, j0, p_rows, p_cols

    def _solve(self, x):
        m, n = x.shape
        xp, nan_rows, nan_cols = self._find_nan_indices(x)
        self._verbose(f"{len(nan_rows)} rows and {len(nan_cols)} columns contain NaNs.")

        if len(nan_rows) == 0:
            return np.arange(m), np.arange(n)
        else:
            rng = np.random.default_rng(seed=self.random_state)
            area_max = -1
            for k in range(self.n_tries):
                area, i0, j0, p_rows, p_cols = self._trial(xp, m, n, rng)
                self._verbose(f"\tTrial {k+1} : submatrix of size {m-i0}x{n-j0} ({area} elements) found.")
                if area > area_max:
                    area_max = area
                    opt = i0, j0, p_rows, p_cols

            i0, j0, p_rows, p_cols = opt
            self._verbose(f"Result: the largest submatrix found is of size {m-i0}x{n-j0} ({area_max} elements) found.")

            rows_to_keep = np.setdiff1d(np.arange(m), nan_rows[p_rows[:i0]])
            cols_to_keep = np.setdiff1d(np.arange(n), nan_cols[p_cols[:j0]])
            return rows_to_keep, cols_to_keep

    def solve(self, X: Union[np.ndarray, pd.DataFrame], return_data: bool = False) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[pd.Index, pd.Index]]:
        """
        Solves the optimal problem of removing NaNs for a 2D array or DataFrame.

        Parameters:
        - X (Union[np.ndarray, pd.DataFrame]): The input 2D array or DataFrame with NaN values.
        - return_data (bool): If True, returns the resulting data; otherwise, returns the indices.

        Returns:
        - Union[Tuple[np.ndarray, np.ndarray], Tuple[pd.Index, pd.Index]]: If return_data is True,
          returns the resulting 2D array or DataFrame; otherwise, returns the indices of rows and columns to retain.

        Raises:
        - ValueError: If the input data is not a numpy array or a pandas DataFrame,
          or if the input numpy array does not have ndim==2,
          or if the OptiMask algorithm encounters an error during optimization.
        """
        if not isinstance(X, (np.ndarray, pd.DataFrame)):
            raise ValueError("Input 'X' must be a numpy array or a pandas DataFrame.")

        if isinstance(X, np.ndarray) and X.ndim != 2:
            raise ValueError("For a numpy array, 'X' must have ndim==2.")

        rows, cols = self._solve(np.isnan(np.asarray(X)))

        if np.isnan(np.asarray(X)[rows][:, cols]).any():
            raise ValueError("The OptiMask algorithm encountered an error.")

        if isinstance(X, pd.DataFrame):
            if return_data:
                return X.iloc[rows, cols].copy()
            else:
                return X.index[rows].copy(), X.columns[cols].copy()

        if isinstance(X, np.ndarray):
            if return_data:
                return X[rows][:, cols].copy()
            else:
                return rows, cols
