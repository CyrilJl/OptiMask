# -*- coding: utf-8 -*-

# author : Cyril Joly


from typing import Tuple, Union

import numpy as np
import pandas as pd


class OptiMask:
    """OptiMask class for finding the largest submatrix without NaN values."""

    @staticmethod
    def _sort_by_na_number(x: np.ndarray, axis: int = 0) -> np.ndarray:
        return np.argsort(x.sum(axis=axis))[::-1]

    @staticmethod
    def _height(x: np.ndarray) -> int:
        return x.nonzero()[0].max() + 1 if np.any(x) else 0

    @classmethod
    def _sort_by_na_max_index(cls, x: np.ndarray, axis: int = 0) -> np.ndarray:
        if axis == 0:
            return np.argsort([cls._height(p) for p in x.T])[::-1]
        if axis == 1:
            return np.argsort([cls._height(p) for p in x])[::-1]

    @classmethod
    def _get_largest_rectangle(cls, x: np.ndarray) -> Tuple[int, int]:
        m, n = x.shape
        heights = [cls._height(_) for _ in x.T]
        areas = [(m - h) * (n - k) for k, h in enumerate(heights)]
        i0 = np.argmax(areas)
        return i0, heights[i0]

    @classmethod
    def _solve(cls, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        x = np.isnan(np.array(x))
        p_cols_1 = cls._sort_by_na_number(x, axis=0)
        p_rows_1 = cls._sort_by_na_number(x, axis=1)
        x = x[p_rows_1][:, p_cols_1]

        p_cols_2 = cls._sort_by_na_max_index(x, axis=0)
        p_rows_2 = cls._sort_by_na_max_index(x[:, p_cols_2], axis=1)

        x = x[p_rows_2][:, p_cols_2]
        i0, j0 = cls._get_largest_rectangle(x)
        return np.sort(p_rows_1[p_rows_2][j0:]), np.sort(p_cols_1[p_cols_2][i0:])

    @classmethod
    def solve(
        cls,
        data: Union[np.ndarray, pd.DataFrame],
        return_data: bool = False,
    ) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[pd.Index, pd.Index], np.ndarray, pd.DataFrame]:
        """
        Finds the indices or submatrix of the largest non-NaN region in the input data.

        Args:
            data (Union[np.ndarray, pd.DataFrame]): Input data matrix or DataFrame.
            return_data (bool, optional): If True, returns the submatrix instead of indices.

        Returns:
            Union[Tuple[np.ndarray, np.ndarray], Tuple[pd.Index, pd.Index], np.ndarray, pd.DataFrame]:
                If 'return_data' is False:
                    - If 'data' is np.ndarray: Tuple of row indices and column indices of the submatrix.
                    - If 'data' is pd.DataFrame: Tuple of row labels and column labels of the submatrix.
                If 'return_data' is True:
                    - If 'data' is np.ndarray: Submatrix of the largest non-NaN region.
                    - If 'data' is pd.DataFrame: SubDataFrame of the largest non-NaN region.
        """
        if not isinstance(data, (np.ndarray, pd.DataFrame)):
            raise ValueError("Input 'data' must be either a numpy array or a pandas DataFrame.")

        # Check dimensions for numpy array
        if isinstance(data, np.ndarray) and data.ndim != 2:
            raise ValueError("For a numpy array, 'data' must have ndim==2.")

        rows, cols = cls._solve(data)

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
