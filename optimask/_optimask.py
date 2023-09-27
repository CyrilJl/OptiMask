# -*- coding: utf-8 -*-

# author : Cyril Joly

from typing import Tuple, Union

import numpy as np
import pandas as pd

__all__ = ["OptiMask"]


class OptiMask:
    """OptiMask computes the largest (non-contiguous) submatrix without NaN values from the input matrix."""
    MAX_STEPS = 32

    @staticmethod
    def _sort_by_na_number(x: np.ndarray, axis: int = 0) -> np.ndarray:
        return np.argsort(x.sum(axis=axis))[::-1]

    @staticmethod
    def _height(x: np.ndarray) -> int:
        return x.nonzero()[0].max() + 1 if np.any(x) else 0

    @classmethod
    def _is_decreasing(cls, x):
        return np.all(x[:-1] >= x[1:])

    @classmethod
    def _sort_by_na_max_index(cls, x: np.ndarray, axis: int = 0) -> np.ndarray:
        if axis == 0:
            return np.argsort([cls._height(p) for p in x.T], kind='mergesort')[::-1]
        if axis == 1:
            return np.argsort([cls._height(p) for p in x], kind='mergesort')[::-1]

    @classmethod
    def _get_largest_rectangle(cls, x: np.ndarray) -> Tuple[int, int]:
        m, n = x.shape
        heights = [cls._height(_) for _ in x.T]
        areas = [(m - h) * (n - k) for k, h in enumerate(heights)]
        i0 = np.argmax(areas)
        return i0, heights[i0]

    @classmethod
    def _is_pareto_ordered(cls, x):
        h0 = np.array([cls._height(_) for _ in x])
        h1 = np.array([cls._height(_) for _ in x.T])
        return cls._is_decreasing(h0) and cls._is_decreasing(h1)

    @classmethod
    def _process_init(cls, x):
        p_cols = cls._sort_by_na_number(x, axis=0)
        p_rows = cls._sort_by_na_number(x, axis=1)
        return x[p_rows][:, p_cols], p_rows, p_cols

    @classmethod
    def _process_step(cls, x, axis):
        p = cls._sort_by_na_max_index(x, axis=axis)
        if axis == 0:
            return x[:, p], p
        if axis == 1:
            return x[p], p

    @classmethod
    def _compute_permutations(cls, x):
        xp, p_cols, p_rows = np.isnan(np.array(x)).copy(), np.arange(x.shape[1]), np.arange(x.shape[0])

        xp, p_rows_step, p_cols_step = cls._process_init(xp)
        p_cols, p_rows = p_cols[p_cols_step], p_rows[p_rows_step]

        step = 0
        while (not cls._is_pareto_ordered(xp)) and (step < cls.MAX_STEPS):
            axis = (step % 2)
            step += 1
            xp, p_step = cls._process_step(xp, axis=axis)
            if axis == 0:
                p_cols = p_cols[p_step]
            if axis == 1:
                p_rows = p_rows[p_step]

        if not cls._is_pareto_ordered(xp):
            raise ValueError("An error occurred during the computation of the optimal permutations.")
        else:
            return xp, p_rows, p_cols

    @classmethod
    def _solve(cls, x):
        xp, p_rows, p_cols = cls._compute_permutations(x)
        i0, j0 = cls._get_largest_rectangle(xp)
        return np.sort(p_rows[j0:]), np.sort(p_cols[i0:])

    @classmethod
    def solve(
        cls,
        data: Union[np.ndarray, pd.DataFrame],
        return_data: bool = False,
    ) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[pd.Index, pd.Index], np.ndarray, pd.DataFrame]:
        """
        Solves the OptiMask problem to find the largest (non-contiguous) submatrix without NaN values in the input data.

        Args:
            data (Union[np.ndarray, pd.DataFrame]): The input data as a numpy array or a pandas DataFrame.
            return_data (bool, optional): If True, returns the submatrix data. If False, returns row and column indices only. Default is False.

        Returns:
            Union[Tuple[np.ndarray, np.ndarray], Tuple[pd.Index, pd.Index], np.ndarray, pd.DataFrame]: A tuple containing the result based on the `return_data` parameter.

        Example:
        ```
        import numpy as np
        from optimask import OptiMask

        # Create a sample numpy array with NaN values
        data = np.array([
            [1, 2, np.nan, 4],
            [np.nan, 6, np.nan, 8],
            [9, 10, 11, 12],
            [13, 14, np.nan, 16],
        ])

        # Instantiate the OptiMask class
        optimask = OptiMask()

        # Solve for the largest submatrix without NaN values
        rows, cols = optimask.solve(data)

        # Print the result
        print("Row indices of the largest submatrix:", rows)
        print("Column indices of the largest submatrix:", cols)

        # Extract the largest submatrix from the original data
        largest_submatrix = data[rows][:, cols]
        print("Largest submatrix:")
        print(largest_submatrix)
        ```
        """
        if not isinstance(data, types=(np.ndarray, pd.DataFrame)):
            raise ValueError("`data` must be a numpy array or a pandas DataFrame.")

        # Check dimensions for numpy array
        if isinstance(data, np.ndarray) and data.ndim != 2:
            raise ValueError("For a numpy array, 'data' must have ndim==2.")

        rows, cols = cls._solve(data)

        if np.isnan(np.asarray(data)[rows][:, cols]).any():
            raise ValueError("The OptiMask algorithm encountered an error.")

        if isinstance(data, pd.DataFrame):
            if return_data:
                return data.iloc[rows, cols].copy()
            else:
                return data.index[rows], data.columns[cols]

        if isinstance(data, np.ndarray):
            if return_data:
                return data[rows][:, cols].copy()
            else:
                return rows, cols
