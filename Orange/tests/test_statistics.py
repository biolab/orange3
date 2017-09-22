import unittest
import warnings
from itertools import chain

import numpy as np
import scipy as sp
from scipy.sparse import csr_matrix, issparse

from Orange.statistics.util import bincount, countnans, contingency, stats, \
    nanmin, nanmax, unique, nanunique, mean, nanmean, digitize, var


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

    def test_nanunique(self):
        x = csr_matrix(np.array([0, 1, 1, np.nan]))
        np.testing.assert_array_equal(
            nanunique(x),
            np.array([0, 1])
        )

    def test_mean(self):
        for X in self.data:
            X_sparse = csr_matrix(X)
            np.testing.assert_array_equal(
                mean(X_sparse),
                np.mean(X))

        with self.assertWarns(UserWarning):
            mean([1, np.nan, 0])

    def test_nanmean(self):
        for X in self.data:
            X_sparse = csr_matrix(X)
            np.testing.assert_array_equal(
                nanmean(X_sparse),
                np.nanmean(X))

    def test_digitize(self):
        for x in self.data:
            x_sparse = csr_matrix(x)
            bins = np.arange(-2, 2)

            x_shape = x.shape
            np.testing.assert_array_equal(
                np.digitize(x.flatten(), bins).reshape(x_shape),
                digitize(x, bins),
                'Digitize fails on dense data'
            )
            np.testing.assert_array_equal(
                np.digitize(x.flatten(), bins).reshape(x_shape),
                digitize(x_sparse, bins),
                'Digitize fails on sparse data'
            )

    def test_digitize_right(self):
        for x in self.data:
            x_sparse = csr_matrix(x)
            bins = np.arange(-2, 2)

            x_shape = x.shape
            np.testing.assert_array_equal(
                np.digitize(x.flatten(), bins, right=True).reshape(x_shape),
                digitize(x, bins, right=True),
                'Digitize fails on dense data'
            )
            np.testing.assert_array_equal(
                np.digitize(x.flatten(), bins, right=True).reshape(x_shape),
                digitize(x_sparse, bins, right=True),
                'Digitize fails on sparse data'
            )

    def test_digitize_1d_array(self):
        """A consistent return shape must be returned for both sparse and dense."""
        x = np.array([0, 1, 1, 0, np.nan, 0, 1])
        x_sparse = csr_matrix(x)
        bins = np.arange(-2, 2)

        x_shape = x.shape
        np.testing.assert_array_equal(
            [np.digitize(x.flatten(), bins).reshape(x_shape)],
            digitize(x, bins),
            'Digitize fails on 1d dense data'
        )
        np.testing.assert_array_equal(
            [np.digitize(x.flatten(), bins).reshape(x_shape)],
            digitize(x_sparse, bins),
            'Digitize fails on 1d sparse data'
        )

    def test_digitize_sparse_zeroth_bin(self):
        # Setup the data so that the '0's will fit into the '0'th bin.
        data = csr_matrix([
            [0, 0, 0, 1, 1, 0, 0, 1, 0],
            [0, 0, 1, 1, 0, 0, 1, 0, 0],
        ])
        bins = np.array([1])
        # Then digitize should return a sparse matrix
        self.assertTrue(issparse(digitize(data, bins)))

    def test_var(self):
        for data in self.data:
            for axis in chain((None,), range(len(data.shape))):
                # Can't use array_equal here due to differences on 1e-16 level
                np.testing.assert_array_almost_equal(
                    var(csr_matrix(data), axis=axis),
                    np.var(data, axis=axis)
                )
