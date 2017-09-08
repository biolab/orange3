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


class TestCountnans(unittest.TestCase):
    def test_dense_array(self):
        dense = np.array([0, 1, 0, 2, 2, np.nan, 1, np.nan, 0, 1])
        sparse = csr_matrix(dense)

        self.assertEqual(countnans(dense), 2)
        self.assertEqual(countnans(sparse), 2)

    def test_sparse_array(self):
        x = csr_matrix([0, 0, 0, 2, 2, np.nan, 1, np.nan, 2, 1])
        self.assertEqual(countnans(x), 2)

    def test_shape_matches_dense_and_sparse_given_array_and_axis_None(self):
        dense = np.array([0, 1, 0, 2, 2, np.nan, 1, np.nan, 0, 1])
        sparse = csr_matrix(dense)
        self.assertEqual(countnans(dense).shape, countnans(sparse).shape)

    def test_shape_matches_dense_and_sparse_given_array_and_axis_0(self):
        dense = np.array([0, 1, 0, 2, 2, np.nan, 1, np.nan, 0, 1])
        sparse = csr_matrix(dense)
        self.assertEqual(countnans(dense, axis=0).shape, countnans(sparse, axis=0).shape)

    def test_shape_matches_dense_and_sparse_given_array_and_axis_1(self):
        dense = np.array([0, 1, 0, 2, 2, np.nan, 1, np.nan, 0, 1])
        sparse = csr_matrix(dense)
        self.assertEqual(countnans(dense, axis=1).shape, countnans(sparse, axis=1).shape)

    def test_2d_matrix(self):
        dense = [[1, np.nan, 1, 2],
                 [2, np.nan, 2, 3]]
        sparse = csr_matrix(dense)
        expected = 2

        self.assertEqual(countnans(dense), expected)
        self.assertEqual(countnans(sparse), expected)

    def test_on_columns(self):
        dense = [[1, np.nan, 1, 2],
                 [2, np.nan, 2, 3]]
        sparse = csr_matrix(dense)
        expected = [0, 2, 0, 0]

        np.testing.assert_equal(countnans(dense, axis=0), expected)
        np.testing.assert_equal(countnans(sparse, axis=0), expected)

    def test_on_rows(self):
        dense = [[1, np.nan, 1, 2],
                 [2, np.nan, 2, 3]]
        sparse = csr_matrix(dense)
        expected = [1, 1]

        np.testing.assert_equal(countnans(dense, axis=1), expected)
        np.testing.assert_equal(countnans(sparse, axis=1), expected)

    def test_1d_weights_with_axis_0(self):
        dense = [[1, 1, np.nan, 1],
                 [np.nan, 1, 1, 1]]
        sparse = csr_matrix(dense)
        w = np.array([0.5, 1, 1, 1])

        np.testing.assert_equal(countnans(dense, w, axis=0), [.5, 0, 1, 0])
        np.testing.assert_equal(countnans(sparse, w, axis=0), [.5, 0, 1, 0])

    def test_1d_weights_with_axis_1(self):
        dense = [[1, 1, np.nan, 1],
                 [np.nan, 1, 1, 1]]
        sparse = csr_matrix(dense)
        w = np.array([0.5, 1])

        np.testing.assert_equal(countnans(dense, w, axis=1), [.5, 1])
        np.testing.assert_equal(countnans(sparse, w, axis=1), [.5, 1])

    def test_2d_weights(self):
        # 2d weights really only matter with dense matrices since we assume
        # sparse matrices are huge, and constructing a full 2d weight matrix
        # seems excessive
        x = [[0, np.nan, 1, 1],
             [0, np.nan, 2, 2]]
        w = np.array([[1, 2, 3, 4],
                      [5, 6, 7, 8]])

        np.testing.assert_equal(countnans(x, weights=w, axis=0), [0, 8, 0, 0])
        np.testing.assert_equal(countnans(x, weights=w, axis=1), [2, 6])


class TestBincount(unittest.TestCase):
    def test_count_nans(self):
        dense = [0, 0, 1, 2, np.nan, 2]
        sparse = csr_matrix(dense)
        expected = 1

        np.testing.assert_equal(bincount(dense)[1], expected)
        np.testing.assert_equal(bincount(sparse)[1], expected)

    def test_adds_empty_bins(self):
        dense = np.array([0, 1, 3, 5])
        sparse = csr_matrix(dense)
        expected = [1, 1, 0, 1, 0, 1]

        np.testing.assert_equal(bincount(dense)[0], expected)
        np.testing.assert_equal(bincount(sparse)[0], expected)

    def test_maxval_adds_empty_bins(self):
        dense = [1, 1, 1, 2, 3, 2]
        sparse = csr_matrix(dense)

        max_val = 5
        expected = [0, 3, 2, 1, 0, 0]

        np.testing.assert_equal(bincount(dense, max_val=max_val)[0], expected)
        np.testing.assert_equal(bincount(sparse, max_val=max_val)[0], expected)
