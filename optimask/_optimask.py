# -*- coding: utf-8 -*-
"""
author: Cyril Joly
"""

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

    def __init__(self, n_tries=10, max_steps=32, random_state=None, verbose=False):
        self.n_tries = n_tries
        self.max_steps = max_steps
        self.random_state = random_state
        self.verbose = bool(verbose)

    def _verbose(self, msg):
        """
        Print verbose information if verbose is True.

        Args:
            msg (str): The message to print.
        """
        if self.verbose:
            print(msg)

    @staticmethod
    def _heights(df, axis=0):
        """
        Calculate heights along the specified axis.

        Args:
            df (pd.DataFrame): The DataFrame.
            axis (int): The axis along which to calculate heights.

        Returns:
            np.ndarray: Heights along the specified axis.
        """
        return df.groupby(axis)[1 - axis].max().values + 1

    @staticmethod
    def _is_decreasing(x) -> bool:
        return all(x[:-1] >= x[1:])

    @classmethod
    def _sort_by_na_max_index(cls, height) -> np.ndarray:
        """
        Sort array indices based on the maximum value of another array in descending order.

        Args:
            height (np.ndarray): The array to sort.

        Returns:
            np.ndarray: Sorted indices.
        """
        return np.argsort(-height, kind='mergesort')

    @classmethod
    def _get_largest_rectangle(cls, heights, m, n):
        areas = (m - heights) * (n - np.arange(len(heights)))
        i0 = np.argmax(areas)
        return i0, heights[i0], areas[i0]

    @staticmethod
    def _permutation_index(p):
        inv = np.empty_like(p)
        inv[p] = np.arange(len(inv), dtype=inv.dtype)
        return inv

    @classmethod
    def _is_pareto_ordered(cls, hx, hy):
        return cls._is_decreasing(hx) and cls._is_decreasing(hy)

    def _trial(self, rng, df, m, n, m_nan, n_nan):
        """
        Perform a single optimization trial.

        Args:
            rng (np.random.Generator): Random number generator.
            df (pd.DataFrame): The DataFrame.
            m (int): Total height of the input array.
            n (int): Total width of the input array.
            m_nan (int): Number of rows with NaN values.
            n_nan (int): Number of columns with NaN values.

        Returns:
            Tuple[int, int, int, np.ndarray, np.ndarray]: Area, indices, and permutations.
        """
        p_rows = rng.permutation(m_nan)
        p_cols = rng.permutation(n_nan)

        df_trial = pd.DataFrame()
        df_trial[0] = self._permutation_index(p_rows)[df[0].values]
        df_trial[1] = self._permutation_index(p_cols)[df[1].values]

        step = 0
        heights = self._heights(df_trial, axis=0), self._heights(df_trial, axis=1)
        while (not self._is_pareto_ordered(*heights)) and (step < self.max_steps):
            axis = (step % 2)
            step += 1
            p_step = self._sort_by_na_max_index(heights[axis])
            df_trial[axis] = self._permutation_index(p_step)[df_trial[axis].values]
            if axis == 0:
                p_rows = p_rows[p_step]
                heights = (heights[0][p_step], self._heights(df_trial, axis=1))
            if axis == 1:
                p_cols = p_cols[p_step]
                heights = (self._heights(df_trial, axis=0), heights[1][p_step])

        if not self._is_pareto_ordered(*heights):
            raise ValueError("An error occurred while calculating optimal permutations. "
                             "You can try again with a larger `max_steps` value.")
        else:
            i0, j0, area = self._get_largest_rectangle(heights[1], m, n)
            return area, i0, j0, p_rows, p_cols

    def _solve(self, x):
        """
        Solve the optimal problem of removing NaNs for a 2D array.

        Args:
            x (np.ndarray): The input 2D array with NaN values.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Rows and columns to retain.
        """
        m, n = x.shape
        df = pd.DataFrame()
        iy, ix = np.isnan(x).nonzero()
        if len(iy)>0:
            rows_with_nan, df[0] = np.unique(iy, return_inverse=True)
            cols_with_nan, df[1] = np.unique(ix, return_inverse=True)
            m_nan, n_nan = len(rows_with_nan), len(cols_with_nan)

            rng = np.random.default_rng(seed=self.random_state)
            area_max = -1
            for k in range(self.n_tries):
                area, i0, j0, p_rows, p_cols = self._trial(rng, df, m, n, m_nan, n_nan)
                self._verbose(f"\tTrial {k + 1} : submatrix of size {m - j0}x{n - i0} ({area} elements) found.")
                if area > area_max:
                    area_max = area
                    opt = i0, j0, p_rows, p_cols

            i0, j0, p_rows, p_cols = opt
            self._verbose(f"Result: the largest submatrix found is of size {m - j0}x{n - i0} ({area_max} elements) found.")

            rows_to_keep = np.setdiff1d(np.arange(m), rows_with_nan[p_rows[:j0]])
            cols_to_keep = np.setdiff1d(np.arange(n), cols_with_nan[p_cols[:i0]])
            return rows_to_keep, cols_to_keep
        else:
            return np.arange(m), np.arange(n)

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
        if not isinstance(X, (np.ndarray, pd.DataFrame)):
            raise ValueError("Input 'X' must be a numpy array or a pandas DataFrame.")

        if isinstance(X, np.ndarray) and X.ndim != 2:
            raise ValueError("For a numpy array, 'X' must have ndim==2.")

        rows, cols = self._solve(np.asarray(X))

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
