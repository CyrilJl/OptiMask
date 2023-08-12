# -*- coding: utf-8 -*-

# author : Cyril Joly


from typing import Tuple, Union
import numpy as np
import pandas as pd


class OptiMask:
    """
    A class for optimizing data masking strategies based on NaN values.

    This class provides methods to optimize the arrangement of NaN values
    in a given data array or DataFrame for efficient analysis or processing.
    The optimization aims to remove rows and columns from the input matrix
    to eliminate NaN values while maximizing the remaining data coverage.

    Attributes:
        None

    Methods:
        height(x: np.ndarray) -> int:
            Calculate the height of the first non-NaN element in each column.

        permutation_columns(x: np.ndarray) -> np.ndarray:
            Get the permutation indices of columns based on the number of NaN values.

        permutation_rows(x: np.ndarray) -> np.ndarray:
            Get the permutation indices of rows based on the number of NaN values.

        _solve(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            Internal method to solve the optimization problem for a given data array.

        solve(data: Union[np.ndarray, pd.DataFrame], return_data: bool = False) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[pd.Index, pd.Index], np.ndarray, pd.DataFrame]:
            Solve the optimization problem for the provided data.

    """

    @staticmethod
    def height(x: np.ndarray) -> int:
        """
        Calculate the height of the first non-NaN element in each column.

        Args:
            x (np.ndarray): Input data array.

        Returns:
            int: The height of the first non-NaN element in each column.
        """
        non_zero_rows = np.nonzero(x)[0]
        return non_zero_rows.max() + 1 if len(non_zero_rows) > 0 else 0

    @staticmethod
    def permutation_columns(x: np.ndarray) -> np.ndarray:
        """
        Get the permutation indices of columns based on the number of NaN values.

        Args:
            x (np.ndarray): Input data array.

        Returns:
            np.ndarray: Permutation indices of columns based on the number of NaN values.
        """
        return np.isnan(x).sum(axis=0).argsort()[::-1]

    @staticmethod
    def permutation_rows(x: np.ndarray) -> np.ndarray:
        """
        Get the permutation indices of rows based on the number of NaN values.

        Args:
            x (np.ndarray): Input data array.

        Returns:
            np.ndarray: Permutation indices of rows based on the number of NaN values.
        """
        mask_nan = np.isnan(x)
        sum_mask = mask_nan.sum(axis=1)

        if np.any(sum_mask):
            sorted_indices = np.argsort(sum_mask)[::-1]
            non_zero_indices = np.nonzero(sum_mask)[0]
            i0 = np.argmax(non_zero_indices >= 1) + 1
            identity = np.arange(i0, len(mask_nan))
            ret = np.concatenate([OptiMask.permutation_rows(
                x[sorted_indices][:i0][:, 1:]), identity])
            return sorted_indices[ret]
        else:
            return np.arange(mask_nan.shape[0])

    @classmethod
    def _solve(cls, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Internal method to solve the optimization problem for a given data array.

        Args:
            x (np.ndarray): Input data array.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Rows and columns indices for optimized arrangement.
        """
        m, n = x.shape
        mask_nan = np.isnan(x)
        p_cols = cls.permutation_columns(x)
        p_rows = cls.permutation_rows(x)
        processed_nan = mask_nan[p_rows][:, p_cols]

        r = np.max(np.flip(np.maximum.accumulate(
            np.flip([cls.height(x) for x in processed_nan.T] + [0]))[1:]))
        r = np.array([[j, i, (n - i) * (m - j)] for i, j in enumerate(r)])
        k = np.argmax(r[:, 2])
        i, j = r[k, [0, 1]]

        rows, cols = np.sort(p_rows[i:]), np.sort(p_cols[j:])
        return rows, cols

    @classmethod
    def solve(cls, data: Union[np.ndarray, pd.DataFrame], return_data: bool = False) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[pd.Index, pd.Index], np.ndarray, pd.DataFrame]:
        """
        Solve the optimization problem for the provided data.

        Args:
            data (Union[np.ndarray, pd.DataFrame]): Input data array or DataFrame.
            return_data (bool, optional): Whether to return the optimized data. Defaults to False.

        Returns:
            Union[Tuple[np.ndarray, np.ndarray], Tuple[pd.Index, pd.Index], np.ndarray, pd.DataFrame]:
                If `return_data` is True:
                    If `data` is a DataFrame, returns a tuple of row and column indices or slices for optimized arrangement.
                    If `data` is a NumPy array, returns the optimized subset of the array.
                If `return_data` is False:
                    If `data` is a DataFrame, returns a tuple of row and column index objects for optimized arrangement.
                    If `data` is a NumPy array, returns a tuple of row and column indices for optimized arrangement.
        """
        if not isinstance(data, (np.ndarray, pd.DataFrame)):
            raise ValueError(
                "Input data must be a NumPy array or a Pandas DataFrame.")

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
