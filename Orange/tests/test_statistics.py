import unittest
import warnings

from functools import wraps
from itertools import chain

import numpy as np
import scipy as sp
from scipy.sparse import csr_matrix, issparse

from Orange.statistics.util import bincount, countnans, contingency, stats, \
    nanmin, nanmax, unique, nanunique, mean, nanmean, digitize, var


def dense_sparse(test_case):
    """Run a single test case on both dense and sparse data."""
    @wraps(test_case)
    def _wrapper(self):
        test_case(self, lambda x: np.array(x))
        test_case(self, lambda x: csr_matrix(x))

    return _wrapper


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

    def test_var(self):
        for data in self.data:
            for axis in chain((None,), range(len(data.shape))):
                # Can't use array_equal here due to differences on 1e-16 level
                np.testing.assert_array_almost_equal(
                    var(csr_matrix(data), axis=axis),
                    np.var(data, axis=axis)
                )


class TestDigitize(unittest.TestCase):
    def setUp(self):
        self.data = [
            np.array([
                [0., 1.,     0., np.nan, 3.,     5.],
                [0., 0., np.nan, np.nan, 5., np.nan],
                [0., 0.,     0., np.nan, 7.,     6.]]),
            np.zeros((2, 3)),
            np.ones((2, 3)),
        ]

    @dense_sparse
    def test_digitize(self, array):
        for x_original in self.data:
            x = array(x_original)
            bins = np.arange(-2, 2)

            x_shape = x.shape
            np.testing.assert_array_equal(
                np.digitize(x_original.flatten(), bins).reshape(x_shape),
                digitize(x, bins),
            )

    @dense_sparse
    def test_digitize_right(self, array):
        for x_original in self.data:
            x = array(x_original)
            bins = np.arange(-2, 2)

            x_shape = x.shape
            np.testing.assert_array_equal(
                np.digitize(x_original.flatten(), bins, right=True).reshape(x_shape),
                digitize(x, bins, right=True)
            )

    @dense_sparse
    def test_digitize_1d_array(self, array):
        """A consistent return shape must be returned for both sparse and dense."""
        x_original = np.array([0, 1, 1, 0, np.nan, 0, 1])
        x = array(x_original)
        bins = np.arange(-2, 2)

        x_shape = x_original.shape
        np.testing.assert_array_equal(
            [np.digitize(x_original.flatten(), bins).reshape(x_shape)],
            digitize(x, bins),
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


class TestCountnans(unittest.TestCase):
    @dense_sparse
    def test_1d_array(self, array):
        x = array([0, 1, 0, 2, 2, np.nan, 1, np.nan, 0, 1])
        self.assertEqual(countnans(x), 2)

    @dense_sparse
    def test_1d_array_with_axis_0(self, array):
        x = array([0, 1, 0, 2, 2, np.nan, 1, np.nan, 0, 1])
        expected = 2

        self.assertEqual(countnans(x, axis=0), expected)

    @dense_sparse
    def test_1d_array_with_axis_1_raises_exception(self, array):
        with self.assertRaises(ValueError):
            countnans(array([0, 1, 0, 2, 2, np.nan, 1, np.nan, 0, 1]), axis=1)

    @dense_sparse
    def test_shape_matches_dense_and_sparse(self, array):
        x = array([[0, 1, 0, 2, 2, np.nan, 1, np.nan, 0, 1],
                   [1, 2, 2, 1, np.nan, 1, 2, 3, np.nan, 3]])
        expected = 4

        self.assertEqual(countnans(x), expected)

    @dense_sparse
    def test_shape_matches_dense_and_sparse_with_axis_0(self, array):
        x = array([[0, 1, 0, 2, 2, np.nan, 1, np.nan, 0, 1],
                   [1, 2, 2, 1, np.nan, 1, 2, np.nan, 3, 3]])
        expected = [0, 0, 0, 0, 1, 1, 0, 2, 0, 0]

        np.testing.assert_equal(countnans(x, axis=0), expected)

    @dense_sparse
    def test_shape_matches_dense_and_sparse_with_axis_1(self, array):
        x = array([[0, 1, 0, 2, 2, np.nan, 1, np.nan, 0, 1],
                   [1, 2, 2, 1, np.nan, 1, 2, 3, np.nan, 3]])
        expected = [2, 2]

        np.testing.assert_equal(countnans(x, axis=1), expected)

    @dense_sparse
    def test_2d_matrix(self, array):
        x = array([[1, np.nan, 1, 2],
                   [2, np.nan, 2, 3]])
        expected = 2

        self.assertEqual(countnans(x), expected)

    @dense_sparse
    def test_on_columns(self, array):
        x = array([[1, np.nan, 1, 2],
                   [2, np.nan, 2, 3]])
        expected = [0, 2, 0, 0]

        np.testing.assert_equal(countnans(x, axis=0), expected)

    @dense_sparse
    def test_on_rows(self, array):
        x = array([[1, np.nan, 1, 2],
                   [2, np.nan, 2, 3]])
        expected = [1, 1]

        np.testing.assert_equal(countnans(x, axis=1), expected)

    @dense_sparse
    def test_1d_weights_with_axis_0(self, array):
        x = array([[1, 1, np.nan, 1],
                   [np.nan, 1, 1, 1]])
        w = np.array([0.5, 1, 1, 1])

        np.testing.assert_equal(countnans(x, w, axis=0), [.5, 0, 1, 0])

    @dense_sparse
    def test_1d_weights_with_axis_1(self, array):
        x = array([[1, 1, np.nan, 1],
                   [np.nan, 1, 1, 1]])
        w = np.array([0.5, 1])

        np.testing.assert_equal(countnans(x, w, axis=1), [.5, 1])

    @dense_sparse
    def test_2d_weights(self, array):
        x = array([[np.nan, np.nan, 1,      1 ],
                   [     0, np.nan, 2, np.nan ]])
        w = np.array([[1, 2, 3, 4],
                      [5, 6, 7, 8]])

        np.testing.assert_equal(countnans(x, w), 17)
        np.testing.assert_equal(countnans(x, w, axis=0), [1, 8, 0, 8])
        np.testing.assert_equal(countnans(x, w, axis=1), [3, 14])

    @dense_sparse
    def test_dtype(self, array):
        x = array([0, np.nan, 2, 3])
        w = np.array([0, 1.5, 0, 0])

        self.assertIsInstance(countnans(x, w, dtype=np.int32), np.int32)
        self.assertEqual(countnans(x, w, dtype=np.int32), 1)
        self.assertIsInstance(countnans(x, w, dtype=np.float64), np.float64)
        self.assertEqual(countnans(x, w, dtype=np.float64), 1.5)


class TestBincount(unittest.TestCase):
    @dense_sparse
    def test_count_nans(self, array):
        x = array([0, 0, 1, 2, np.nan, 2])
        expected = 1

        np.testing.assert_equal(bincount(x)[1], expected)

    @dense_sparse
    def test_adds_empty_bins(self, array):
        x = array([0, 1, 3, 5])
        expected = [1, 1, 0, 1, 0, 1]

        np.testing.assert_equal(bincount(x)[0], expected)

    @dense_sparse
    def test_maxval_adds_empty_bins(self, array):
        x = array([1, 1, 1, 2, 3, 2])
        max_val = 5
        expected = [0, 3, 2, 1, 0, 0]

        np.testing.assert_equal(bincount(x, max_val=max_val)[0], expected)

    @dense_sparse
    def test_maxval_doesnt_truncate_values_when_too_small(self, array):
        x = array([1, 1, 1, 2, 3, 2])
        max_val = 1
        expected = [0, 3, 2, 1]

        np.testing.assert_equal(bincount(x, max_val=max_val)[0], expected)

    @dense_sparse
    def test_minlength_adds_empty_bins(self, array):
        x = array([1, 1, 1, 2, 3, 2])
        minlength = 5
        expected = [0, 3, 2, 1, 0]

        np.testing.assert_equal(bincount(x, minlength=minlength)[0], expected)

    @dense_sparse
    def test_weights(self, array):
        x = array([0, 0, 1, 1, 2, 2, 3, 3])
        w = np.array([1, 2, 0, 0, 1, 1, 0, 1])

        expected = [3, 0, 2, 1]
        np.testing.assert_equal(bincount(x, w)[0], expected)

    @dense_sparse
    def test_weights_with_nans(self, array):
        x = array([0, 0, 1, 1, np.nan, 2, np.nan, 3])
        w = np.array([1, 2, 0, 0, 1, 1, 0, 1])

        expected = [3, 0, 1, 1]
        np.testing.assert_equal(bincount(x, w)[0], expected)
