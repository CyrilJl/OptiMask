from typing import Tuple, Union

import numpy as np
import pandas as pd
from . import optimask_rust
from ._misc import (
    EmptyInputError,
    InvalidDimensionError,
    OptiMaskAlgorithmError,
    check_params,
    warning,
)

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

    def _verbose(self, msg):
        if self.verbose:
            warning(msg)

    @staticmethod
    def groupby_max(a, b, n):
        # Inputs a, b are np.ndarray (uint32). PyO3 converts to Vec<u32>.
        # optimask_rust.groupby_max returns a Python list.
        return np.array(optimask_rust.groupby_max(a, b, n), dtype=np.uint32)

    @staticmethod
    def is_decreasing(h):
        # Input h is np.ndarray (uint32). PyO3 converts to Vec<u32>.
        # optimask_rust.is_decreasing returns a bool.
        return optimask_rust.is_decreasing(h)

    @classmethod
    def apply_permutation(cls, p: np.ndarray, x: np.ndarray, inplace: bool):
        # p and x are uint32 NumPy arrays.
        # Rust functions numba_apply_permutation and numba_apply_permutation_inplace
        # take Vec<u32> (PyO3 handles conversion) and return Vec<u32> (Python list).
        if inplace:
            if not isinstance(x, np.ndarray):
                raise TypeError("Input 'x' for inplace permutation must be a NumPy array.")
            # The rust function numba_apply_permutation_inplace takes p and x (as Vec<u32>)
            # and returns the permuted x as a new Vec<u32> (Python list).
            permuted_x_list = optimask_rust.numba_apply_permutation_inplace(p, x.astype(np.uint32)) # Ensure uint32 for Rust
            x[:] = permuted_x_list  # Update original array content
            # No return for in-place operations typically
        else:
            permuted_x_list = optimask_rust.numba_apply_permutation(p.astype(np.uint32), x.astype(np.uint32)) # Ensure uint32
            return np.array(permuted_x_list, dtype=np.uint32) # Return new NumPy array

    @staticmethod
    def apply_p_step(p_step, a, b):
        # Inputs p_step, a, b are np.ndarray (uint32). PyO3 converts to Vec<u32>.
        # optimask_rust.apply_p_step returns a tuple of Python lists.
        res_a, res_b = optimask_rust.apply_p_step(p_step.astype(np.uint32), a.astype(np.uint32), b.astype(np.uint32)) # Ensure uint32
        return np.array(res_a, dtype=np.uint32), np.array(res_b, dtype=np.uint32)

    @staticmethod
    def _get_largest_rectangle(heights, m, n):
        areas = (m - heights) * (n - np.arange(len(heights)))
        i0 = np.argmax(areas)
        return i0, heights[i0], areas[i0]

    @staticmethod
    def _preprocess(x):
        """
        Preprocesses the input array to identify rows and columns containing NaNs.

        Args:
            x (np.ndarray): The input 2D array with NaN values.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
                - iy, ix = np.isnan(x).nonzero()
                - rows_with_nan: Rows that contain NaNs.
                - cols_with_nan: Columns that contain NaNs.
        """
        m, n = x.shape
        # Input x is np.ndarray. Rust takes PyReadonlyArray2<f64>.
        # Ensure x is float64.
        # optimask_rust._preprocess returns a tuple of Python lists.
        # x is already np.asarray from solve() method
        iy_l, ix_l, rwn_l, cwn_l = optimask_rust._preprocess(x.astype(np.float64))
        return (
            np.array(iy_l, dtype=np.uint32),
            np.array(ix_l, dtype=np.uint32),
            np.array(rwn_l, dtype=np.uint32),
            np.array(cwn_l, dtype=np.uint32),
        )

    def _trial(self, k, rng, m_nan, n_nan, iy, ix, m, n):
        if k:
            p_rows = rng.permutation(m_nan).astype(np.uint32)
            p_cols = rng.permutation(n_nan).astype(np.uint32)
            iy_trial = self.apply_permutation(p_rows, iy, inplace=False)
            ix_trial = self.apply_permutation(p_cols, ix, inplace=False)
        else:
            p_rows = np.arange(m_nan, dtype=np.uint32)
            p_cols = np.arange(n_nan, dtype=np.uint32)
            iy_trial = iy.copy()
            ix_trial = ix.copy()

        hy = self.groupby_max(iy_trial, ix_trial, m_nan)
        step = 0
        is_pareto_ordered = False
        while not is_pareto_ordered and step < self.max_steps:
            axis = step % 2
            step += 1
            if axis == 0:
                p_step = (-hy).argsort(kind="mergesort").astype(np.uint32)
                self.apply_permutation(p_step, iy_trial, inplace=True)
                p_rows, hy = self.apply_p_step(p_step, p_rows, hy)
                hx = self.groupby_max(ix_trial, iy_trial, n_nan)
                is_pareto_ordered = self.is_decreasing(hx)
            else:
                p_step = (-hx).argsort(kind="mergesort").astype(np.uint32)
                self.apply_permutation(p_step, ix_trial, inplace=True)
                hy = self.groupby_max(iy_trial, ix_trial, m_nan)
                p_cols, hx = self.apply_p_step(p_step, p_cols, hx)
                is_pareto_ordered = self.is_decreasing(hy)

        if not is_pareto_ordered:
            raise OptiMaskAlgorithmError(
                "An error occurred while calculating optimal permutations. "
                "You can try again with a larger `max_steps` value."
            )
        else:
            i0, j0, area = self._get_largest_rectangle(hx, m, n)
            return area, i0, j0, p_rows, p_cols

    @staticmethod
    def compute_to_keep(size, index_with_nan, permutation, split):
        """
        Computes the indices to keep after removing a subset of indices with NaNs.

        Args:
            size (int): The total number of indices.
            index_with_nan (np.ndarray): The indices that contain NaNs.
            permutation (np.ndarray): The permutation array.
            split (int): The split point in the permutation array.

        Returns:
            np.ndarray: The indices to keep after removing the subset with NaNs.
        """
        # Inputs index_with_nan, permutation are np.ndarray (uint32). PyO3 converts to Vec<u32>.
        # size and split are Python integers.
        # optimask_rust.compute_to_keep returns a Python list.
        return np.array(
            optimask_rust.compute_to_keep(size, index_with_nan.astype(np.uint32), permutation.astype(np.uint32), split), # Ensure uint32
            dtype=np.uint32,
        )

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

        if len(iy) == m * n:
            self._verbose("The array is full of NaNs.")
            if m <= n:
                return np.array([]), np.arange(n)
            else:
                return np.arange(m), np.array([], dtype=np.uint32)

        if len(rows_with_nan) == 1:
            if n - n_nan <= n_nan * (m - m_nan):
                return np.setdiff1d(np.arange(m), rows_with_nan), np.arange(n)
            else:
                return np.arange(m), np.setdiff1d(np.arange(n), cols_with_nan)

        if len(cols_with_nan) == 1:
            if m - m_nan <= m_nan * (n - n_nan):
                return np.arange(m), np.setdiff1d(np.arange(n), cols_with_nan)
            else:
                return np.setdiff1d(np.arange(m), rows_with_nan), np.arange(n)

        else:
            rng = np.random.default_rng(seed=self.random_state)
            area_max = -1
            for k in range(self.n_tries):
                area, i0, j0, p_rows, p_cols = self._trial(k, rng, m_nan, n_nan, iy, ix, m, n)
                self._verbose(f"\tTrial {k + 1} : submatrix of size {m - j0}x{n - i0} ({area} elements) found.")
                if area > area_max:
                    area_max = area
                    opt = i0, j0, p_rows, p_cols

            i0, j0, p_rows, p_cols = opt
            self._verbose(
                f"Result: the largest submatrix found is of size {m - j0}x{n - i0} ({area_max} elements) found."
            )

            rows_to_keep = self.compute_to_keep(size=m, index_with_nan=rows_with_nan, permutation=p_rows, split=j0)
            cols_to_keep = self.compute_to_keep(size=n, index_with_nan=cols_with_nan, permutation=p_cols, split=i0)
            return rows_to_keep, cols_to_keep

    @staticmethod
    def has_nan_in_subset(X, rows, cols):
        """
        Checks if there are any NaN values in the specified subset of the array.

        Args:
            X (np.ndarray): The input 2D array.
            rows (np.ndarray): The row indices of the subset.
            cols (np.ndarray): The column indices of the subset.

        Returns:
            bool: True if there are NaN values in the subset, False otherwise.
        """
        # Input X is np.ndarray (float64). Rust takes PyReadonlyArray2<f64>.
        # Inputs rows, cols are np.ndarray. Rust takes Vec<usize>.
        # X is already np.asarray from solve() method. Ensure it's float64 for Rust.
        # PyO3 should handle conversion of NumPy int arrays (like uint32, uint64, int64) to Vec<usize>.
        return optimask_rust.has_nan_in_subset(X.astype(np.float64), rows, cols)

    def solve(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        return_data: bool = False,
        check_result=False,
    ) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[pd.Index, pd.Index]]:
        """
        Solves the optimal problem of removing NaNs for a 2D array or DataFrame.

        Args:
            X (Union[np.ndarray, pd.DataFrame]): The input 2D array or DataFrame with NaN values.
            return_data (bool): If True, returns the resulting data; otherwise, returns the indices.
            check_result (bool): If True, checks if the computed submatrix contains NaNs, for tests purposes.
            Disabled by default as it can slow down the computation and the algorithm has proven to be reliable.

        Returns:
            Union[Tuple[np.ndarray, np.ndarray], Tuple[pd.Index, pd.Index]]: If return_data is True, returns
            the resulting 2D array or DataFrame; otherwise, returns the indices of rows and columns to retain.

        Raises:
            InvalidDimensionError: If the input numpy array does not have ndim==2.
            EmptyInputError: If the input data is empty.
            OptiMaskAlgorithmError: If the OptiMask algorithm encounters an error during optimization.
            ValueError: If the input DataFrame's index contains non-unique entries.
        """
        try:
            import polars as pl

            has_polars = True
        except ImportError:
            has_polars = False

        if has_polars:
            check_params(X, types=(np.ndarray, pd.DataFrame, pl.DataFrame))
        else:
            check_params(X, types=(np.ndarray, pd.DataFrame))

        if isinstance(X, np.ndarray) and X.ndim != 2:
            raise InvalidDimensionError("For a numpy array, 'X' must have ndim == 2.")

        if X.shape[0] == 0 or X.shape[1] == 0:
            raise EmptyInputError("`X` is empty.")

        rows, cols = self._solve(np.asarray(X))

        if check_result and self.has_nan_in_subset(np.asarray(X), rows, cols):
            raise OptiMaskAlgorithmError(
                "The OptiMask algorithm encountered an error, computed submatrix contains NaNs."
            )

        if isinstance(X, pd.DataFrame):
            if return_data:
                return X.iloc[rows, cols].copy()
            else:
                if not X.index.is_unique:
                    raise ValueError("The index contains non-unique entries!")
                return X.index[rows].copy(), X.columns[cols].copy()

        elif isinstance(X, np.ndarray):
            if return_data:
                return X[np.ix_(rows, cols)].copy()
            else:
                return rows, cols

        elif has_polars and isinstance(X, pl.DataFrame):
            if return_data:
                return X[rows][:, cols]
            else:
                return rows, cols
