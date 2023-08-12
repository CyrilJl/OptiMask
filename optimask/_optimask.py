# -*- coding: utf-8 -*-

# author : Cyril Joly


from typing import Tuple, Union

import numpy as np
import pandas as pd


class OptiMask:
    """
    This class is used to find the set of rows and columns to remove from a 2D array (numpy array or pandas DataFrame) with missing data, in order to obtain the resulting submatrix without any holes and as large as possible (i.e., with the maximum number of cells).

    The algorithm utilizes a method based on sorting rows and columns based on the number of missing values (NaN) they contain. Then, it searches for the optimal permutation of rows and columns to obtain the desired submatrix.

    Reference: https://mathematica.stackexchange.com/a/284918/92680

    Example usage:
    >>> from airpy.maths.optimask import OptiMask
    >>> import numpy as np

    >>> m, n = 120, 20
    >>> data = np.zeros((m, n))
    >>> data[24:72, 2] = np.nan
    >>> data[0:24, 5] = np.nan
    >>> data[100, :] = np.nan
    >>> data[80:90, 7] = np.nan

    >>> rows, cols = OptiMask()(data)
    >>> subdata = OptiMask()(data, return_data=True)

    Note:
        This class can be used to enhance model efficiency by removing missing data while retaining as much information as possible.
    """

    @classmethod
    def height(cls, x: np.ndarray) -> int:
        r = x.nonzero()[0]
        if len(r) > 0:
            return r.max() + 1
        else:
            return 0

    @staticmethod
    def permutation_columns(x: np.ndarray) -> np.ndarray:
        return np.isnan(np.array(x)).sum(axis=0).argsort()[::-1]

    @classmethod
    def permutation_rows(cls, x: np.ndarray) -> np.ndarray:
        m = np.isnan(np.array(x))
        if np.sum(m) > 0:
            if x.shape[1] > 1:
                p = m.sum(axis=1).argsort()[::-1]
                i0 = np.argmax((m[p].sum(axis=1) > 0).nonzero()) + 1
                identity = np.arange(i0, len(m))
                ret = np.concatenate(
                    [cls.permutation_rows(x[p][:i0][:, 1:]), identity])
                return p[ret]
            else:
                return m.sum(axis=0).argsort()[::-1]
        else:
            return np.arange(m.shape[0])

    @classmethod
    def solve(cls, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        m, n = x.shape
        mask_nan = np.isnan(np.array(x))
        p_cols = cls.permutation_columns(np.array(x))
        p_rows = cls.permutation_rows(np.array(x))
        processed_nan = mask_nan[p_rows][:, p_cols]
        r = [cls.height(x) for x in processed_nan.T] + [0]
        r = np.flip([np.max(np.flip(r)[:k + 1]) for k in range(1, len(r))])
        r = np.array([[j, i, (n - i) * (m - j)] for i, j in enumerate(r)])
        k = np.argmax(r[:, 2])
        i, j = r[k, [0, 1]]

        rows, cols = np.sort(p_rows[i:]), np.sort(p_cols[j:])
        return rows, cols

    @classmethod
    def __call__(cls, data: Union[np.ndarray, pd.DataFrame], return_data: bool = False) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[pd.Index, pd.Index], np.ndarray, pd.DataFrame]:
        """Calls the solve method to find the rows and columns to remove and returns the results as indices or a submatrix.

        Parameters:
            data (np.ndarray or pd.DataFrame): The 2D array with missing data.
            return_data (bool, optional): Indicates whether the resulting submatrix should be returned. Default is False.

        Returns:
            Union[Tuple[np.ndarray, np.ndarray], Tuple[pd.Index, pd.Index], np.ndarray, pd.DataFrame]: The indices of rows and columns to remove or the resulting submatrix, based on the value of return_data.
        """
        rows, cols = cls.solve(data)

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
