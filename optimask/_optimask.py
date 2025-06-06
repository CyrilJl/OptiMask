from typing import Tuple, Union, List, Optional, Any
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from numba import bool_, njit, prange, uint32
from numba.types import UniTuple

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
        n_jobs (int): Number of parallel jobs to run for trials. Default is 1.
                      If -1, then the number of jobs is set to the number of CPUs.
    """

    def __init__(self, n_tries=5, max_steps=16, random_state=None, verbose=False, n_jobs=1):
        self.n_tries = n_tries
        self.max_steps = max_steps
        self.random_state = random_state
        self.verbose = bool(verbose)
        self.n_jobs = n_jobs

    def _verbose(self, msg):
        if self.verbose:
            warning(msg)

    @staticmethod
    @njit(uint32[:](uint32[:], uint32[:], uint32), boundscheck=False, cache=True)
    def groupby_max(a, b, n):
        """
        numba equivalent to :
            size_a = len(a)
            ret = np.zeros(n, dtype=np.uint32)
            np.maximum.at(ret, a, b + 1)
            return ret
        """
        size_a = len(a)
        ret = np.zeros(n, dtype=np.uint32)
        for k in range(size_a):
            ak = a[k]
            ret[ak] = max(ret[ak], b[k] + 1)
        return ret

    @staticmethod
    @njit(bool_(uint32[:]), boundscheck=False, cache=True)
    def is_decreasing(h):
        for i in range(len(h) - 1):
            if h[i] < h[i + 1]:
                return False
        return True

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
        """
        Applies a permutation to an array.

        Args:
            p (np.ndarray): The permutation array.
            x (np.ndarray): The array to be permuted.
            inplace (bool): If True, applies the permutation in place; otherwise, returns a new permuted array.

        Returns:
            np.ndarray: The permuted array if inplace is False; otherwise, None.
        """
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
        iy, ix = np.empty(m * n, dtype=np.uint32), np.empty(m * n, dtype=np.uint32)
        cols_index_mapper = -np.ones(n, dtype=np.int32)
        rows_with_nan = np.empty(m, dtype=np.uint32)
        n_rows_with_nan = 0
        n_cols_with_nan = 0
        cnt = 0
        for i in range(m):
            row_has_nan = False
            for j in range(n):
                if np.isnan(x[i, j]):
                    row_has_nan = True
                    iy[cnt] = n_rows_with_nan

                    if cols_index_mapper[j] >= 0:
                        ix[cnt] = cols_index_mapper[j]
                    else:
                        ix[cnt] = n_cols_with_nan
                        cols_index_mapper[j] = n_cols_with_nan
                        n_cols_with_nan += 1
                    cnt += 1

            if row_has_nan:
                rows_with_nan[n_rows_with_nan] = i
                n_rows_with_nan += 1

        iy, ix = iy[:cnt], ix[:cnt]
        rows_with_nan = rows_with_nan[:n_rows_with_nan]
        cols_with_nan = np.flatnonzero(cols_index_mapper >= 0)[
            cols_index_mapper[cols_index_mapper >= 0].argsort()
        ].astype(np.uint32)
        return iy, ix, rows_with_nan, cols_with_nan

    def _trial(self, k_and_seed, m_nan, n_nan, iy, ix, m, n):
        k, trial_seed = k_and_seed
        if k:
            rng = np.random.default_rng(trial_seed)
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
    @njit(uint32[:](uint32, uint32[:], uint32[:], uint32), parallel=True, boundscheck=False, cache=True)
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
        mask = np.zeros(size, dtype=np.uint8)
        for i in prange(split):
            mask[index_with_nan[permutation[i]]] = 1

        count = size - split

        result = np.empty(count, dtype=np.uint32)
        idx = 0
        for i in range(size):
            if mask[i] == 0:
                result[idx] = i
                idx += 1
        return result

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
            # Determine the number of workers for parallel execution
            if self.n_jobs == -1:
                actual_n_workers = os.cpu_count()
                if actual_n_workers is None:  # os.cpu_count() can return None
                    actual_n_workers = 1
            elif self.n_jobs <= 0:  # Treat 0 or negative (not -1) as 1 worker
                actual_n_workers = 1
            else:
                actual_n_workers = self.n_jobs

            if self.n_tries > 0:
                n_workers_for_pool = min(actual_n_workers, self.n_tries)
            else: # No trials, so no workers needed for pool.
                n_workers_for_pool = 0

            # Prepare arguments for each trial, including seeds
            trial_k_and_seeds: List[Tuple[int, Any]] = []
            if self.random_state is not None:
                ss = np.random.SeedSequence(self.random_state)
                child_seeds = ss.spawn(self.n_tries)
                for k_idx in range(self.n_tries):
                    if k_idx == 0: # Trial k=0 doesn't use RNG for permutation
                        trial_k_and_seeds.append((k_idx, None))
                    else:
                        trial_k_and_seeds.append((k_idx, child_seeds[k_idx]))
            else:
                for k_idx in range(self.n_tries):
                    trial_k_and_seeds.append((k_idx, None)) # default_rng(None) is random

            results = []
            if n_workers_for_pool > 1 and self.n_tries > 0:
                with ThreadPoolExecutor(max_workers=n_workers_for_pool) as executor:
                    futures = [
                        executor.submit(self._trial, k_s, m_nan, n_nan, iy, ix, m, n)
                        for k_s in trial_k_and_seeds
                    ]
                    for future in futures:
                        results.append(future.result())
            else: # Sequential execution
                for k_s in trial_k_and_seeds:
                    result = self._trial(k_s, m_nan, n_nan, iy, ix, m, n)
                    results.append(result)

            area_max = -1
            opt: Optional[Tuple[np.uint32, np.uint32, np.ndarray, np.ndarray]] = None

            if not results and self.n_tries > 0:
                # This case implies all trials failed, which is unexpected if _trial always returns.
                # Or if n_tries was > 0 but trial_k_and_seeds was empty (should not happen).
                raise OptiMaskAlgorithmError("Optimization failed: No results returned from trials.")

            for k_idx, result_tuple in enumerate(results):
                area, i0, j0, p_rows, p_cols = result_tuple
                original_k = trial_k_and_seeds[k_idx][0] # Get the original k value for logging
                self._verbose(f"\tTrial {original_k + 1} : submatrix of size {m - j0}x{n - i0} ({area} elements) found.")
                if area > area_max:
                    area_max = area
                    opt = (i0, j0, p_rows, p_cols)

            if opt is None:
                if self.n_tries == 0:
                    # No trials were run, return all rows/cols as no optimization was performed.
                    # This behavior is debatable: either this or raise error.
                    # For now, returning all seems like a safe default for "0 tries".
                    # However, the original code would error here due to `opt` not being assigned.
                    # To maintain consistency with potential original error for n_tries=0:
                    # Let's ensure `opt` related error can still occur if it's truly uninitialized AND expected.
                    # The problem with original code is `opt` might not be defined. Here it *is* defined as None.
                    # If n_tries = 0, we should probably return m, n earlier or handle it.
                    # Given the problem focuses on parallelizing, stick to that.
                    # If `opt` is still None and `n_tries` > 0, it's an issue.
                    # The `if not results and self.n_tries > 0` check above should catch this.
                    # If `n_tries` is 0, `results` is empty, `opt` is `None`.
                    # The code proceeds to `i0, j0, p_rows, p_cols = opt` which will fail.
                    # This matches the original code's behavior if n_tries=0.
                    pass # Let it fail on `i0,j0... = opt` if n_tries = 0
                else: # n_tries > 0 but opt is still None
                    raise OptiMaskAlgorithmError(
                        "Optimization failed: No optimal solution found after all trials."
                    )


            i0, j0, p_rows, p_cols = opt # This line will raise TypeError if opt is None
            self._verbose(
                f"Result: the largest submatrix found is of size {m - j0}x{n - i0} ({area_max} elements) found."
            )

            rows_to_keep = self.compute_to_keep(size=m, index_with_nan=rows_with_nan, permutation=p_rows, split=j0)
            cols_to_keep = self.compute_to_keep(size=n, index_with_nan=cols_with_nan, permutation=p_cols, split=i0)
            return rows_to_keep, cols_to_keep

    @staticmethod
    @njit(boundscheck=False, cache=True)
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
        for i in range(rows.size):
            for j in range(cols.size):
                if np.isnan(X[rows[i], cols[j]]):
                    return True
        return False

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
