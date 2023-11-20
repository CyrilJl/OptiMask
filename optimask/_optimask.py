# -*- coding: utf-8 -*-

# author : Cyril Joly

from typing import Tuple, Union

import numpy as np
import pandas as pd

__all__ = ["OptiMask"]


class OptiMask:
    """
    OptiMask est une classe permettant de calculer les lignes et les colonnes optimales à conserver dans un tableau
    ou un DataFrame 2D afin de supprimer les valeurs NaN et de conserver le nombre maximal de cellules non-NaN.
    La classe utilise une approche d'optimisation heuristique, et augmenter la valeur de `n_tries` conduit généralement
    à de meilleurs résultats, pouvant atteindre ou approcher étroitement la quantité optimale.

    Paramètres :
    - n_tries (int) : Le nombre de tentatives d'optimisation. Des valeurs plus élevées peuvent conduire à de meilleurs résultats.
    - max_steps (int) : Le nombre maximum d'étapes à effectuer dans chaque tentative d'optimisation.
    - random_state (Union[int, None]) : Graine pour le générateur de nombres aléatoires.

    Attributs :
    - n_tries (int) : Le nombre de tentatives d'optimisation.
    - max_steps (int) : Le nombre maximum d'étapes à effectuer dans chaque tentative d'optimisation.
    - random_state (Union[int, None]) : Graine pour le générateur de nombres aléatoires.
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

    @classmethod
    def _heights(cls, x, axis=0):
        return np.argmax(np.cumsum(x, axis=axis), axis=axis) + 1

    @staticmethod
    def _is_decreasing(x) -> bool:
        return np.all(x[:-1] >= x[1:])

    @classmethod
    def _sort_by_na_max_index(cls, height) -> np.ndarray:
        return np.argsort(-height, kind='mergesort')

    @classmethod
    def _get_largest_rectangle(cls, x, m, n):
        heights = cls._heights(x, axis=0)
        areas = [(m - h) * (n - k) for k, h in enumerate(heights)]
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
            if axis == 1:
                p_rows = p_rows[p_step]

            heights = self._heights(xpp, axis=0), self._heights(xpp, axis=1)

        if not self._is_pareto_ordered(*heights):
            raise ValueError("Une erreur s'est produite lors du calcul des permutations optimales. Vous pouvez réessayer avec une valeur `max_steps` plus grande.")
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
        if len(nan_rows) == 0:
            return np.arange(m), np.arange(n)
        else:
            rng = np.random.default_rng(seed=self.random_state)

            area_max = -1
            for _ in range(self.n_tries):
                area, i0, j0, p_rows, p_cols = self._trial(xp, m, n, rng)
                if area > area_max:
                    area_max = area
                    opt = i0, j0, p_rows, p_cols

            i0, j0, p_rows, p_cols = opt
            rows_to_remove, cols_to_remove = nan_rows[p_rows[:i0]
                                                      ], nan_cols[p_cols[:j0]]

            rows_to_keep = np.array(
                [_ for _ in range(m) if _ not in rows_to_remove])
            cols_to_keep = np.array(
                [_ for _ in range(n) if _ not in cols_to_remove])
            return rows_to_keep, cols_to_keep

    def solve(self, X: Union[np.ndarray, pd.DataFrame], return_data: bool = False) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[pd.Index, pd.Index]]:
        """
        Résoud le problème optimal de suppression des NaN pour un tableau 2D ou un DataFrame.

        Paramètres :
        - X (Union[np.ndarray, pd.DataFrame]) : Le tableau 2D ou le DataFrame d'entrée avec des valeurs NaN.
        - return_data (bool) : Si True, renvoie les données résultantes ; sinon, renvoie les indices.

        Renvoie :
        - Union[Tuple[np.ndarray, np.ndarray], Tuple[pd.Index, pd.Index]] : Si return_data est True,
          renvoie le tableau 2D ou le DataFrame résultant ; sinon, renvoie les indices des lignes et des colonnes à conserver.

        Lève :
        - ValueError : Si les données d'entrée ne sont pas un tableau numpy ou un DataFrame pandas,
          ou si le tableau numpy d'entrée n'a pas ndim==2,
          ou si l'algorithme rencontre une erreur pendant l'optimisation.
        """
        if not isinstance(X, (np.ndarray, pd.DataFrame)):
            raise ValueError()

        if isinstance(X, np.ndarray) and X.ndim != 2:
            raise ValueError("Pour un tableau numpy, 'X' doit avoir ndim==2.")

        rows, cols = self._solve(np.isnan(np.asarray(X)))

        if np.isnan(np.asarray(X)[rows][:, cols]).any():
            raise ValueError("L'algorithme OptiMask a rencontré une erreur.")

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
