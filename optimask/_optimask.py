# -*- coding: utf-8 -*-

# author : Cyril Joly

from typing import Tuple, Union

import numpy as np
import pandas as pd

__all__ = ["OptiMask"]


class OptiMask:
    """
    OptiMask is a class for computing the optimal rows and columns to keep in a 2D array or DataFrame
    in order to remove NaN values and retain the maximum amount of non-NaN cells. The class employs a heuristic
    optimization approach, and increasing the value of `n_tries` generally results in more remaining cells,
    potentially reaching or closely approaching the optimal amount.

    Parameters:
    - n_tries (int): The number of optimization attempts. Higher values may lead to better results.
    - max_steps (int): The maximum number of steps to perform in each optimization attempt.
    - random_state (Union[int, None]): Seed for the random number generator.

    Attributes:
    - n_tries (int): The number of optimization attempts.
    - max_steps (int): The maximum number of steps to perform in each optimization attempt.
    - random_state (Union[int, None]): Seed for the random number generator.
    """

    def __init__(self, n_tries=10, max_steps=32, random_state=None):
        self.n_tries = n_tries
        self.max_steps = max_steps
        self.random_state = random_state

    @staticmethod
    def _find_nan_indices(x):
        nan_rows, nan_cols = np.nonzero(x)
        nan_rows = np.unique(nan_rows)
        nan_cols = np.unique(nan_cols)
        return x[nan_rows][:, nan_cols], nan_rows, nan_cols

    @staticmethod
    def _height(x: np.ndarray) -> int:
        return x.nonzero()[0].max() + 1 if np.any(x) else 0

    @staticmethod
    def _is_decreasing(x):
        return np.all(x[:-1] >= x[1:])

    @classmethod
    def _sort_by_na_max_index(cls, x: np.ndarray, axis: int = 0) -> np.ndarray:
        if axis == 0:
            return np.argsort([cls._height(p) for p in x.T], kind='mergesort')[::-1]
        if axis == 1:
            return np.argsort([cls._height(p) for p in x], kind='mergesort')[::-1]

    @classmethod
    def _get_largest_rectangle(cls, x, m, n):
        heights = [cls._height(_) for _ in x.T]
        areas = [(m - h) * (n - k) for k, h in enumerate(heights)]
        i0 = np.argmax(areas)
        return heights[i0], i0, areas[i0]

    @classmethod
    def _is_pareto_ordered(cls, x):
        h0 = np.array([cls._height(_) for _ in x])
        h1 = np.array([cls._height(_) for _ in x.T])
        return cls._is_decreasing(h0) and cls._is_decreasing(h1)

    @classmethod
    def _process_step(cls, x, axis):
        p = cls._sort_by_na_max_index(x, axis=axis)
        if axis == 0:
            return x[:, p], p
        if axis == 1:
            return x[p], p

    @staticmethod
    def _random_start(xp, rng):
        p_rows = rng.permutation(xp.shape[0])
        p_cols = rng.permutation(xp.shape[1])
        return xp[p_rows][:, p_cols].copy(), p_rows, p_cols

    def _compute_permutations(self, xpp, p_rows, p_cols):
        step = 0
        while (not self._is_pareto_ordered(xpp)) and (step < self.max_steps):
            axis = (step % 2)
            step += 1
            xpp, p_step = self._process_step(xpp, axis=axis)
            if axis == 0:
                p_cols = p_cols[p_step]
            if axis == 1:
                p_rows = p_rows[p_step]

        if not self._is_pareto_ordered(xpp):
            raise ValueError("An error occurred during the computation of the optimal permutations.")
        else:
            return xpp, p_rows, p_cols

    def _trial(self, xp, m, n, rng):
        xpp, p_rows, p_cols = self._random_start(xp=xp, rng=rng)
        xpp, p_rows, p_cols = self._compute_permutations(xpp, p_rows, p_cols)
        i0, j0, area = self._get_largest_rectangle(xpp, m, n)
        return area, i0, j0, p_rows, p_cols

    def _solve(self, x):
        m, n = x.shape
        xp, nan_rows, nan_cols = self._find_nan_indices(x)
        rng = np.random.default_rng(seed=self.random_state)

        area_max = -1
        for _ in range(self.n_tries):
            area, i0, j0, p_rows, p_cols = self._trial(xp, m, n, rng)
            if area > area_max:
                area_max = area
                opt = i0, j0, p_rows, p_cols

        i0, j0, p_rows, p_cols = opt
        rows_to_remove, cols_to_remove = nan_rows[p_rows[:i0]], nan_cols[p_cols[:j0]]

        rows_to_keep = np.array([_ for _ in range(m) if _ not in rows_to_remove])
        cols_to_keep = np.array([_ for _ in range(n) if _ not in cols_to_remove])
        return rows_to_keep, cols_to_keep

    def solve(self, X: Union[np.ndarray, pd.DataFrame], return_data: bool = False) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[pd.Index, pd.Index]]:
        """
        Solve the optimal NaN removal problem for a 2D array or DataFrame.

        Parameters:
        - X (Union[np.ndarray, pd.DataFrame]): The input 2D array or DataFrame with NaN values.
        - return_data (bool): If True, return the resulting data; otherwise, return indices.

        Returns:
        - Union[Tuple[np.ndarray, np.ndarray], Tuple[pd.Index, pd.Index]]: If return_data is True,
          returns the resulting 2D array or DataFrame; otherwise, returns the indices of rows and columns to keep.

        Raises:
        - ValueError: If the input data is not a numpy array or a pandas DataFrame,
          or if the input numpy array does not have ndim==2,
          or if the algorithm encounters an error during optimization.
        """
        if not isinstance(X, (np.ndarray, pd.DataFrame)):
            raise ValueError("`data` must be a numpy array or a pandas DataFrame.")

        # Check dimensions for numpy array
        if isinstance(X, np.ndarray) and X.ndim != 2:
            raise ValueError("For a numpy array, 'X' must have ndim==2.")

        rows, cols = self._solve(np.isnan(np.asarray(X)))

        if np.isnan(np.asarray(X)[rows][:, cols]).any():
            raise ValueError("The OptiMask algorithm encountered an error.")

        if isinstance(X, pd.DataFrame):
            if return_data:
                return X.iloc[rows, cols].copy()
            else:
                return X.index[rows], data.columns[cols]

        if isinstance(X, np.ndarray):
            if return_data:
                return X[rows][:, cols].copy()
            else:
                return rows, cols
