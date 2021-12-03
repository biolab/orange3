# pylint: disable=no-self-use
import time
import unittest
import warnings
from itertools import chain
from functools import partial, wraps

import numpy as np
from scipy.sparse import csr_matrix, issparse, lil_matrix, csc_matrix, \
    SparseEfficiencyWarning

from Orange.data.util import assure_array_dense
from Orange.statistics.util import bincount, countnans, contingency, digitize, \
    mean, nanmax, nanmean, nanmedian, nanmin, nansum, nanunique, stats, std, \
    unique, var, nanstd, nanvar, nanmode, nan_to_num, FDR, isnan, any_nan, \
    all_nan
from sklearn.utils import check_random_state


def dense_sparse(test_case):
    # type: (Callable) -> Callable
    """Run a single test case on both dense and sparse data."""
    @wraps(test_case)
    def _wrapper(self):

        def sparse_with_explicit_zero(x, array):
            """Inject one explicit zero into a sparse array."""
            np_array, sp_array = np.atleast_2d(x), array(x)
            assert issparse(sp_array), 'Can not inject explicit zero into non-sparse matrix'

            zero_indices = np.argwhere(np_array == 0)
            if zero_indices.size:
                with warnings.catch_warnings():
                    # this is just inefficiency in tests, not the tested code
                    warnings.filterwarnings("ignore", ".*",
                                            SparseEfficiencyWarning)
                    sp_array[tuple(zero_indices[0])] = 0

            return sp_array

        # Make sure to call setUp and tearDown methods in between test runs so
        # any widget state doesn't interfere between tests
        def _setup_teardown():
            self.tearDown()
            self.setUp()

        test_case(self, np.array)
        _setup_teardown()
        test_case(self, csr_matrix)
        _setup_teardown()
        test_case(self, csc_matrix)
        _setup_teardown()
        test_case(self, partial(sparse_with_explicit_zero, array=csr_matrix))
        _setup_teardown()
        test_case(self, partial(sparse_with_explicit_zero, array=csc_matrix))

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
        x = np.array([0, 1, 0, 2, np.nan, np.nan])
        y = np.array([0, 0, 1, np.nan, 0, np.nan])
        cont_table, col_nans, row_nans, nans = contingency(x, y, 2, 2)
        np.testing.assert_equal(cont_table, [[1, 1, 0],
                                             [1, 0, 0],
                                             [0, 0, 0]])
        np.testing.assert_equal(col_nans, [1, 0, 0])
        np.testing.assert_equal(row_nans, [0, 0, 1])
        self.assertEqual(1, nans)

    def test_weighted_contingency(self):
        x = np.array([0, 1, 0, 2, np.nan, np.nan])
        y = np.array([0, 0, 1, np.nan, 0, np.nan])
        w = np.array([1, 2, 2, 3, 4, 2])
        cont_table, col_nans, row_nans, nans = contingency(
            x, y, 2, 2, weights=w)
        np.testing.assert_equal(cont_table, [[1, 2, 0],
                                             [2, 0, 0],
                                             [0, 0, 0]])
        np.testing.assert_equal(col_nans, [4, 0, 0])
        np.testing.assert_equal(row_nans, [0, 0, 3])
        self.assertEqual(2, nans)

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
            ["", "a", np.nan, 0],
            ["a", "", np.nan, 1],
            ["a", "b", 0, 0],
        ], dtype=object)
        np.testing.assert_equal(stats(X), [[np.inf, -np.inf, 0, 0, 1, 2],
                                           [np.inf, -np.inf, 0, 0, 1, 2],
                                           [np.inf, -np.inf, 0, 0, 2, 1],
                                           [np.inf, -np.inf, 0, 0, 0, 3]])

    def test_stats_long_string_mem_use(self):
        X = np.full((1000, 1000), "a", dtype=object)
        t = time.time()
        stats(X)
        t_a = time.time() - t  # time for an array with constant-len strings

        # Add one very long string
        X[0, 0] = "a"*2000

        # The implementation of stats() in Orange 3.30.2 used .astype("str")
        # internally. X.astype("str") would take ~1000x the memory as X,
        # because its type would be "<U1000" (the length of the longest string).
        # That is about 7.5 GiB of memory on a 64-bit Linux system

        # Because it is hard to measure CPU, we here measure time as
        # memory allocation of such big tables takes time. On Marko's
        # Linux system .astype("str") took ~3 seconds.
        t = time.time()
        stats(X)
        t_b = time.time() - t
        self.assertLess(t_b, 2*t_a + 0.1)  # some grace period

    def test_nanmin_nanmax(self):
        warnings.filterwarnings("ignore", r".*All-NaN slice encountered.*")
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

    @dense_sparse
    def test_nansum(self, array):
        for X in self.data:
            X_sparse = array(X)
            np.testing.assert_array_equal(
                nansum(X_sparse),
                np.nansum(X))

    def test_mean(self):
        # This warning is not unexpected and it's correct
        warnings.filterwarnings("ignore", r".*mean\(\) resulted in nan.*")
        for X in self.data:
            X_sparse = csr_matrix(X)
            np.testing.assert_array_equal(
                mean(X_sparse),
                np.mean(X))

        with self.assertWarns(UserWarning):
            mean([1, np.nan, 0])

    def test_nanmode(self):
        X = np.array([[np.nan, np.nan, 1, 1],
                      [2, np.nan, 1, 1]])
        mode, count = nanmode(X, 0)
        np.testing.assert_array_equal(mode, [[2, np.nan, 1, 1]])
        np.testing.assert_array_equal(count, [[1, np.nan, 2, 2]])
        mode, count = nanmode(X, 1)
        np.testing.assert_array_equal(mode, [[1], [1]])
        np.testing.assert_array_equal(count, [[2], [2]])

    @dense_sparse
    def test_nanmedian(self, array):
        for X in self.data:
            X_sparse = array(X)
            np.testing.assert_array_equal(
                nanmedian(X_sparse),
                np.nanmedian(X))

    @dense_sparse
    def test_nanmedian_more_nonzeros(self, array):
        X = np.ones((10, 10))
        X[:5, 0] = np.nan
        X[:6, 1] = 0
        X_sparse = array(X)
        np.testing.assert_array_equal(
            nanmedian(X_sparse),
            np.nanmedian(X)
        )

    def test_var(self):
        for data in self.data:
            for axis in chain((None,), range(len(data.shape))):
                # Can't use array_equal here due to differences on 1e-16 level
                np.testing.assert_array_almost_equal(
                    var(csr_matrix(data), axis=axis),
                    np.var(data, axis=axis)
                )

    def test_var_with_ddof(self):
        x = np.random.uniform(0, 10, (20, 100))
        for axis in [None, 0, 1]:
            np.testing.assert_almost_equal(
                np.var(x, axis=axis, ddof=10),
                var(csr_matrix(x), axis=axis, ddof=10),
            )

    @dense_sparse
    def test_nanvar(self, array):
        for X in self.data:
            X_sparse = array(X)
            np.testing.assert_almost_equal(
                nanvar(X_sparse),
                np.nanvar(X), decimal=14)  # np.nanvar and bn.nanvar differ slightly

    def test_nanvar_with_ddof(self):
        x = np.random.uniform(0, 10, (20, 100))
        np.fill_diagonal(x, np.nan)
        for axis in [None, 0, 1]:
            np.testing.assert_almost_equal(
                np.nanvar(x, axis=axis, ddof=10),
                nanvar(csr_matrix(x), axis=axis, ddof=10),
            )

    def test_std(self):
        for data in self.data:
            for axis in chain((None,), range(len(data.shape))):
                # Can't use array_equal here due to differences on 1e-16 level
                np.testing.assert_array_almost_equal(
                    std(csr_matrix(data), axis=axis),
                    np.std(data, axis=axis)
                )

    def test_std_with_ddof(self):
        x = np.random.uniform(0, 10, (20, 100))
        for axis in [None, 0, 1]:
            np.testing.assert_almost_equal(
                np.std(x, axis=axis, ddof=10),
                std(csr_matrix(x), axis=axis, ddof=10),
            )

    @dense_sparse
    def test_nanstd(self, array):
        for X in self.data:
            X_sparse = array(X)
            np.testing.assert_array_equal(
                nanstd(X_sparse),
                np.nanstd(X))

    def test_nanstd_with_ddof(self):
        x = np.random.uniform(0, 10, (20, 100))
        for axis in [None, 0, 1]:
            np.testing.assert_almost_equal(
                np.nanstd(x, axis=axis, ddof=10),
                nanstd(csr_matrix(x), axis=axis, ddof=10),
            )

    def test_FDR(self):
        p_values = np.array([0.0002, 0.0004, 0.00001, 0.0003, 0.0001])
        np.testing.assert_almost_equal(
            np.array([0.00033, 0.0004, 0.00005, 0.00038, 0.00025]),
            FDR(p_values), decimal=5)

    def test_FDR_dependent(self):
        p_values = np.array([0.0002, 0.0004, 0.00001, 0.0003, 0.0001])
        np.testing.assert_almost_equal(
            np.array([0.00076, 0.00091, 0.00011, 0.00086, 0.00057]),
            FDR(p_values, dependent=True), decimal=5)

    def test_FDR_m(self):
        p_values = np.array([0.0002, 0.0004, 0.00001, 0.0003, 0.0001])
        np.testing.assert_almost_equal(
            np.array([0.0002, 0.00024, 0.00003, 0.000225, 0.00015]),
            FDR(p_values, m=3), decimal=5)

    def test_FDR_no_values(self):
        self.assertIsNone(FDR(None))
        self.assertIsNone(FDR([]))
        self.assertIsNone(FDR([0.0002, 0.0004], m=0))

    def test_FDR_list(self):
        p_values = [0.0002, 0.0004, 0.00001, 0.0003, 0.0001]
        result = FDR(p_values)
        self.assertIsInstance(result, list)
        np.testing.assert_almost_equal(
            np.array([0.00033, 0.0004, 0.00005, 0.00038, 0.00025]),
            result, decimal=5)


class TestNanmean(unittest.TestCase):
    def setUp(self):
        self.random_state = check_random_state(42)
        self.x = self.random_state.uniform(size=(10, 5))
        np.fill_diagonal(self.x, np.nan)

    @dense_sparse
    def test_axis_none(self, array):
        np.testing.assert_almost_equal(
            np.nanmean(self.x), nanmean(array(self.x))
        )

    @dense_sparse
    def test_axis_0(self, array):
        np.testing.assert_almost_equal(
            np.nanmean(self.x, axis=0), nanmean(array(self.x), axis=0)
        )

    @dense_sparse
    def test_axis_1(self, array):
        np.testing.assert_almost_equal(
            np.nanmean(self.x, axis=1), nanmean(array(self.x), axis=1)
        )


class TestDigitize(unittest.TestCase):
    def setUp(self):
        # pylint: disable=bad-whitespace
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
        # pylint: disable=bad-whitespace
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

    # object arrays cannot be converted to sparse, so test only for dense
    def test_count_nans_objectarray(self):
        x = np.array([0, 0, 1, 2, np.nan, 2], dtype=object)
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

    @dense_sparse
    def test_weights_with_transposed_x(self, array):
        x = array([0, 0, 1, 1, 2, 2, 3, 3]).T
        w = np.array([1, 2, 0, 0, 1, 1, 0, 1])

        expected = [3, 0, 2, 1]
        np.testing.assert_equal(bincount(x, w)[0], expected)

    @dense_sparse
    def test_all_nans(self, array):
        x = array([np.nan] * 5)
        expected = []

        np.testing.assert_equal(bincount(x)[0], expected)

    @dense_sparse
    def test_all_zeros_or_nans(self, array):
        """Sparse arrays with only nans with no explicit zeros will have no non
        zero indices. Check that this counts the zeros properly."""
        x = array([np.nan] * 5 + [0] * 5)
        expected = [5]

        np.testing.assert_equal(bincount(x)[0], expected)


class TestUnique(unittest.TestCase):
    @dense_sparse
    def test_returns_unique_values(self, array):
        # pylint: disable=bad-whitespace
        x = array([[-1., 1., 0., 2., 3., np.nan],
                   [ 0., 0., 0., 3., 5.,    42.],
                   [-1., 0., 0., 1., 7.,     6.]])
        expected = [-1, 0, 1, 2, 3, 5, 6, 7, 42, np.nan]

        np.testing.assert_equal(unique(x, return_counts=False), expected)

    @dense_sparse
    def test_returns_counts(self, array):
        # pylint: disable=bad-whitespace
        x = array([[-1., 1., 0., 2., 3., np.nan],
                   [ 0., 0., 0., 3., 5.,    42.],
                   [-1., 0., 0., 1., 7.,     6.]])
        expected = [-1, 0, 1, 2, 3, 5, 6, 7, 42, np.nan]
        expected_counts = [2, 6, 2, 1, 2, 1, 1, 1, 1, 1]

        vals, counts = unique(x, return_counts=True)

        np.testing.assert_equal(vals, expected)
        np.testing.assert_equal(counts, expected_counts)

    def test_sparse_explicit_zeros(self):
        # Use `lil_matrix` to fix sparse warning for matrix construction
        x = lil_matrix(np.eye(3))
        x[0, 1] = 0
        x[1, 0] = 0
        x = x.tocsr()
        # Test against identity matrix
        y = csr_matrix(np.eye(3))

        np.testing.assert_array_equal(
            unique(y, return_counts=True),
            unique(x, return_counts=True),
        )

    @dense_sparse
    def test_nanunique_ignores_nans_in_values(self, array):
        # pylint: disable=bad-whitespace
        x = array([[-1., 1., 0., 2., 3., np.nan],
                   [ 0., 0., 0., 3., 5., np.nan],
                   [-1., 0., 0., 1., 7.,     6.]])
        expected = [-1, 0, 1, 2, 3, 5, 6, 7]

        np.testing.assert_equal(nanunique(x, return_counts=False), expected)

    @dense_sparse
    def test_nanunique_ignores_nans_in_counts(self, array):
        # pylint: disable=bad-whitespace
        x = array([[-1., 1., 0., 2., 3., np.nan],
                   [ 0., 0., 0., 3., 5., np.nan],
                   [-1., 0., 0., 1., 7.,     6.]])
        expected = [2, 6, 2, 1, 2, 1, 1, 1]

        np.testing.assert_equal(nanunique(x, return_counts=True)[1], expected)


class TestNanToNum(unittest.TestCase):
    @dense_sparse
    def test_converts_invalid_values(self, array):
        x = np.array([
            [np.nan, 0, 2, np.inf],
            [-np.inf, np.nan, 1e-18, 2],
            [np.nan, np.nan, np.nan, np.nan],
            [np.inf, np.inf, np.inf, np.inf],
        ])
        result = nan_to_num(array(x))
        np.testing.assert_equal(assure_array_dense(result), np.nan_to_num(x))

    @dense_sparse
    def test_preserves_valid_values(self, array):
        x = np.arange(12).reshape((3, 4))
        result = nan_to_num(array(x))
        np.testing.assert_equal(assure_array_dense(result), x)
        np.testing.assert_equal(assure_array_dense(result), np.nan_to_num(x))


class TestIsnan(unittest.TestCase):
    def setUp(self) -> None:
        # pylint: disable=bad-whitespace
        self.x = np.array([
            [0., 1.,     0., np.nan, 3.,     5.],
            [0., 0., np.nan, np.nan, 5., np.nan],
            [0., 0.,     0., np.nan, 7.,     6.],
            [0., 0.,     0.,     5., 7.,     6.],
        ])

    @dense_sparse
    def test_functionality(self, array):
        expected = np.isnan(self.x)
        result = isnan(array(self.x))
        np.testing.assert_equal(assure_array_dense(result), expected)

    @dense_sparse
    def test_out(self, array):
        x = array(self.x)
        x_dtype = x.dtype
        result = isnan(x, out=x)
        self.assertIs(result, x)
        self.assertEqual(x_dtype, result.dtype)


class TestAnyNans(unittest.TestCase):
    def setUp(self) -> None:
        # pylint: disable=bad-whitespace
        self.x_with_nans = np.array([
            [0., 1.,     0., np.nan, 3.,     5.],
            [0., 0., np.nan, np.nan, 5., np.nan],
            [0., 0.,     0., np.nan, 7.,     6.],
            [0., 0.,     0.,     5., 7.,     6.],
        ])
        self.x_no_nans = np.arange(12).reshape((3, 4))

    @dense_sparse
    def test_axis_none_without_nans(self, array):
        self.assertFalse(any_nan(array(self.x_no_nans)))

    @dense_sparse
    def test_axis_none_with_nans(self, array):
        self.assertTrue(any_nan(array(self.x_with_nans)))

    @dense_sparse
    def test_axis_0_without_nans(self, array):
        expected = np.array([0, 0, 0, 0], dtype=bool)
        result = any_nan(array(self.x_no_nans), axis=0)
        np.testing.assert_equal(result, expected)

    @dense_sparse
    def test_axis_0_with_nans(self, array):
        expected = np.array([0, 0, 1, 1, 0, 1], dtype=bool)
        result = any_nan(array(self.x_with_nans), axis=0)
        np.testing.assert_equal(result, expected)

    @dense_sparse
    def test_axis_1_without_nans(self, array):
        expected = np.array([0, 0, 0], dtype=bool)
        result = any_nan(array(self.x_no_nans), axis=1)
        np.testing.assert_equal(result, expected)

    @dense_sparse
    def test_axis_1_with_nans(self, array):
        expected = np.array([1, 1, 1, 0], dtype=bool)
        result = any_nan(array(self.x_with_nans), axis=1)
        np.testing.assert_equal(result, expected)


class TestAllNans(unittest.TestCase):
    def setUp(self) -> None:
        # pylint: disable=bad-whitespace
        self.x_with_nans = np.array([
            [0.,     1.,     0.,     np.nan, 3.,     5.],
            [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            [0.,     0.,     0.,     np.nan, 7.,     6.],
            [0.,     0.,     0.,     np.nan, 7.,     np.nan],
        ])
        self.x_no_nans = np.arange(12).reshape((3, 4))
        self.x_only_nans = (np.ones(12) * np.nan).reshape((3, 4))

    @dense_sparse
    def test_axis_none_without_nans(self, array):
        self.assertFalse(all_nan(array(self.x_no_nans)))

    @dense_sparse
    def test_axis_none_with_nans(self, array):
        self.assertTrue(all_nan(array(self.x_only_nans)))

    @dense_sparse
    def test_axis_0_without_nans(self, array):
        expected = np.array([0, 0, 0, 0], dtype=bool)
        result = all_nan(array(self.x_no_nans), axis=0)
        np.testing.assert_equal(result, expected)

    @dense_sparse
    def test_axis_0_with_nans(self, array):
        expected = np.array([0, 0, 0, 1, 0, 0], dtype=bool)
        result = all_nan(array(self.x_with_nans), axis=0)
        np.testing.assert_equal(result, expected)

    @dense_sparse
    def test_axis_1_without_nans(self, array):
        expected = np.array([0, 0, 0], dtype=bool)
        result = all_nan(array(self.x_no_nans), axis=1)
        np.testing.assert_equal(result, expected)

    @dense_sparse
    def test_axis_1_with_nans(self, array):
        expected = np.array([0, 1, 0, 0], dtype=bool)
        result = all_nan(array(self.x_with_nans), axis=1)
        np.testing.assert_equal(result, expected)


class TestNanModeFixedInScipy(unittest.TestCase):

    @unittest.expectedFailure
    def test_scipy_nanmode_still_wrong(self):
        import scipy.stats
        X = np.array([[np.nan, np.nan, 1, 1],
                      [2, np.nan, 1, 1]])
        mode, count = scipy.stats.mode(X, 0)
        np.testing.assert_array_equal(mode, [[2, np.nan, 1, 1]])
        np.testing.assert_array_equal(count, [[1, np.nan, 2, 2]])
        mode, count = scipy.stats.mode(X, 1)
        np.testing.assert_array_equal(mode, [[1], [1]])
        np.testing.assert_array_equal(count, [[2], [2]])
        # When Scipy's scipy.stats.mode works correcly, remove Orange.statistics.util.nanmode
        # and this test. Also update requirements.
