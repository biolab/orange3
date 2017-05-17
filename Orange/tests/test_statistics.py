import unittest
import warnings
import numpy as np
import scipy as sp
from scipy.sparse import csr_matrix

from Orange.statistics.util import bincount, countnans, contingency, stats, \
    nanmin, nanmax, unique, mean, nanmean


class TestUtil(unittest.TestCase):
    def setUp(self):
        nan = float('nan')
        self.data = [
            np.array([
                [0., 1., 0., nan, 3., 5.],
                [0., 0., nan, nan, 5., nan],
                [0., 0., 0., nan, 7., 6.]]),
            np.zeros((2, 3)),
            np.ones((2, 3)),
        ]

    def test_bincount(self):
        hist, n_nans = bincount([0., 1., np.nan, 3])
        self.assertEqual(n_nans, 1)
        np.testing.assert_equal(hist, [1, 1, 0, 1])

        hist, n_nans = bincount([0., 1., 3], max_val=3)
        self.assertEqual(n_nans, 0)
        np.testing.assert_equal(hist, [1, 1, 0, 1])

    def test_countnans(self):
        np.testing.assert_equal(countnans([[1, np.nan],
                                           [2, np.nan]], axis=0), [0, 2])

    def test_contingency(self):
        x = np.array([0, 1, 0, 2, np.nan])
        y = np.array([0, 0, 1, np.nan, 0])
        cont_table, nans = contingency(x, y, 2, 2)
        np.testing.assert_equal(cont_table, [[1, 1, 0],
                                             [1, 0, 0],
                                             [0, 0, 0]])
        np.testing.assert_equal(nans, [1, 0, 0])

    def test_stats(self):
        X = np.arange(4).reshape(2, 2).astype(float)
        X[1, 1] = np.nan
        np.testing.assert_equal(stats(X), [[0, 2, 1, 0, 0, 2],
                                           [1, 1, 1, 0, 1, 1]])
        # empty table should return ~like metas
        X = X[:0]
        np.testing.assert_equal(stats(X), [[np.inf, -np.inf, 0, 0, 0, 0],
                                           [np.inf, -np.inf, 0, 0, 0, 0]])

    def test_stats_sparse(self):
        X = csr_matrix(np.identity(5))
        np.testing.assert_equal(stats(X), [[0, 1, .2, 0, 4, 1],
                                           [0, 1, .2, 0, 4, 1],
                                           [0, 1, .2, 0, 4, 1],
                                           [0, 1, .2, 0, 4, 1],
                                           [0, 1, .2, 0, 4, 1]])

        # assure last two columns have just zero elements
        X = X[:3]
        np.testing.assert_equal(stats(X), [[0, 1, 1/3, 0, 2, 1],
                                           [0, 1, 1/3, 0, 2, 1],
                                           [0, 1, 1/3, 0, 2, 1],
                                           [0, 0, 0, 0, 3, 0],
                                           [0, 0, 0, 0, 3, 0]])

    def test_stats_weights(self):
        X = np.arange(4).reshape(2, 2).astype(float)
        weights = np.array([1, 3])
        np.testing.assert_equal(stats(X, weights), [[0, 2, 1.5, 0, 0, 2],
                                                    [1, 3, 2.5, 0, 0, 2]])

        X = np.arange(4).reshape(2, 2).astype(object)
        np.testing.assert_equal(stats(X, weights), stats(X))

    def test_stats_weights_sparse(self):
        X = np.arange(4).reshape(2, 2).astype(float)
        X = csr_matrix(X)
        weights = np.array([1, 3])
        np.testing.assert_equal(stats(X, weights), [[0, 2, 1.5, 0, 1, 1],
                                                    [1, 3, 2.5, 0, 0, 2]])

    def test_stats_non_numeric(self):
        X = np.array([
            ['', 'a', 'b'],
            ['a', '', 'b'],
            ['a', 'b', ''],
        ], dtype=object)
        np.testing.assert_equal(stats(X), [[np.inf, -np.inf, 0, 0, 1, 2],
                                           [np.inf, -np.inf, 0, 0, 1, 2],
                                           [np.inf, -np.inf, 0, 0, 1, 2]])

    def test_nanmin_nanmax(self):
        for X in self.data:
            X_sparse = csr_matrix(X)
            for axis in [None, 0, 1]:
                np.testing.assert_array_equal(
                    nanmin(X, axis=axis),
                    np.nanmin(X, axis=axis))

                np.testing.assert_array_equal(
                    nanmin(X_sparse, axis=axis),
                    np.nanmin(X, axis=axis))

                np.testing.assert_array_equal(
                    nanmax(X, axis=axis),
                    np.nanmax(X, axis=axis))

                np.testing.assert_array_equal(
                    nanmax(X_sparse, axis=axis),
                    np.nanmax(X, axis=axis))

    def test_unique(self):
        for X in self.data:
            X_sparse = csr_matrix(X)
            np.testing.assert_array_equal(
                unique(X_sparse, return_counts=False),
                np.unique(X, return_counts=False))

            for a1, a2 in zip(unique(X_sparse, return_counts=True),
                              np.unique(X, return_counts=True)):
                np.testing.assert_array_equal(a1, a2)

    def test_unique_explicit_zeros(self):
        x1 = csr_matrix(np.eye(3))
        x2 = csr_matrix(np.eye(3))

        # set some of-diagonal to explicit zeros
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore",
                                    category=sp.sparse.SparseEfficiencyWarning)
            x2[0, 1] = 0
            x2[1, 0] = 0

        np.testing.assert_array_equal(
            unique(x1, return_counts=False),
            unique(x2, return_counts=False),
        )
        np.testing.assert_array_equal(
            unique(x1, return_counts=True),
            unique(x2, return_counts=True),
        )

    def test_mean(self):
        for X in self.data:
            X_sparse = csr_matrix(X)
            np.testing.assert_array_equal(
                mean(X_sparse),
                np.mean(X))

    def test_nanmean(self):
        for X in self.data:
            X_sparse = csr_matrix(X)
            np.testing.assert_array_equal(
                nanmean(X_sparse),
                np.nanmean(X))
