import os
import unittest
import warnings

import numpy as np
import scipy.sparse as sp

from Orange.util import export_globals, flatten, deprecated, try_, deepgetattr, \
    OrangeDeprecationWarning
from Orange.data import Table
from Orange.data.util import vstack, hstack, array_equal
from Orange.statistics.util import stats
from Orange.tests.test_statistics import dense_sparse
from Orange.util import wrap_callback, get_entry_point

SOMETHING = 0xf00babe


class TestUtil(unittest.TestCase):
    def test_get_entry_point(self):
        # pylint: disable=import-outside-toplevel
        from Orange.canvas.__main__ import main as real_main
        main = get_entry_point("Orange3", "gui_scripts", "orange-canvas")
        self.assertIs(main, real_main)

    def test_export_globals(self):
        self.assertEqual(sorted(export_globals(globals(), __name__)),
                         ['SOMETHING', 'TestUtil'])

    def test_flatten(self):
        self.assertEqual(list(flatten([[1, 2], [3]])), [1, 2, 3])

    def test_deprecated(self):
        @deprecated
        def identity(x):
            return x

        with self.assertWarns(DeprecationWarning) as cm:
            x = identity(10)
        self.assertEqual(x, 10)
        self.assertTrue('deprecated' in cm.warning.args[0])
        self.assertTrue('identity' in cm.warning.args[0])

    def test_try_(self):
        self.assertTrue(try_(lambda: np.ones(3).any()))
        self.assertFalse(try_(lambda: np.whatever()))
        self.assertEqual(try_(len, default=SOMETHING), SOMETHING)

    def test_reprable(self):
        from Orange.data import ContinuousVariable
        from Orange.preprocess.impute import ReplaceUnknownsRandom
        from Orange.statistics.distribution import Continuous
        from Orange.classification import LogisticRegressionLearner

        var = ContinuousVariable('x')
        transform = ReplaceUnknownsRandom(var, Continuous(1, var))

        self.assertEqual(repr(transform).replace('\n', '').replace(' ', ''),
                         "ReplaceUnknownsRandom("
                         "variable=ContinuousVariable(name='x',number_of_decimals=3),"
                         "distribution=Continuous([[0.],[0.]]))")

        # GH 2275
        logit = LogisticRegressionLearner()
        for _ in range(2):
            self.assertEqual(repr(logit), 'LogisticRegressionLearner()')

    def test_deepgetattr(self):
        class a:
            l = []
        self.assertTrue(deepgetattr(a, 'l.__len__.__call__'), a.l.__len__.__call__)
        self.assertTrue(deepgetattr(a, 'l.__nx__.__x__', 42), 42)
        self.assertRaises(AttributeError, lambda: deepgetattr(a, 'l.__nx__.__x__'))

    def test_vstack(self):
        numpy = np.array([[1., 2.], [3., 4.]])
        csr = sp.csr_matrix(numpy)
        csc = sp.csc_matrix(numpy)

        self.assertCorrectArrayType(
            vstack([numpy, numpy]),
            shape=(4, 2), sparsity="dense")
        self.assertCorrectArrayType(
            vstack([csr, numpy]),
            shape=(4, 2), sparsity="sparse")
        self.assertCorrectArrayType(
            vstack([numpy, csc]),
            shape=(4, 2), sparsity="sparse")
        self.assertCorrectArrayType(
            vstack([csc, csr]),
            shape=(4, 2), sparsity="sparse")

    def test_hstack(self):
        numpy = np.array([[1., 2.], [3., 4.]])
        csr = sp.csr_matrix(numpy)
        csc = sp.csc_matrix(numpy)

        self.assertCorrectArrayType(
            hstack([numpy, numpy]),
            shape=(2, 4), sparsity="dense")
        self.assertCorrectArrayType(
            hstack([csr, numpy]),
            shape=(2, 4), sparsity="sparse")
        self.assertCorrectArrayType(
            hstack([numpy, csc]),
            shape=(2, 4), sparsity="sparse")
        self.assertCorrectArrayType(
            hstack([csc, csr]),
            shape=(2, 4), sparsity="sparse")

    def assertCorrectArrayType(self, array, shape, sparsity):
        self.assertEqual(array.shape, shape)
        self.assertEqual(["dense", "sparse"][sp.issparse(array)], sparsity)

    @unittest.skipUnless(os.environ.get('ORANGE_DEPRECATIONS_ERROR'),
                         'ORANGE_DEPRECATIONS_ERROR not set')
    def test_raise_deprecations(self):
        with self.assertRaises(OrangeDeprecationWarning):
            warnings.warn('foo', OrangeDeprecationWarning)

    def test_stats_sparse(self):
        """
        Stats should not fail when trying to calculate mean on sparse data.
        GH-2357
        """
        data = Table("iris")
        sparse_x = sp.csr_matrix(data.X)
        self.assertTrue(stats(data.X).all() == stats(sparse_x).all())

    @dense_sparse
    def test_array_equal(self, array):
        a1 = array([[0., 2.], [3., np.nan]])
        a2 = array([[0., 2.], [3., np.nan]])
        self.assertTrue(array_equal(a1, a2))

        a3 = np.array([[0., 2.], [3., np.nan]])
        self.assertTrue(array_equal(a1, a3))
        self.assertTrue(array_equal(a3, a1))

    @dense_sparse
    def test_array_not_equal(self, array):
        a1 = array([[0., 2.], [3., np.nan]])
        a2 = array([[0., 2.], [4., np.nan]])
        self.assertFalse(array_equal(a1, a2))

        a3 = array([[0., 2.], [3., np.nan], [4., 5.]])
        self.assertFalse(array_equal(a1, a3))

    def test_csc_array_equal(self):
        a1 = sp.csc_matrix(([1, 4, 5], ([0, 0, 1], [0, 2, 2])), shape=(2, 3))
        a2 = sp.csc_matrix(([5, 1, 4], ([1, 0, 0], [2, 0, 2])), shape=(2, 3))
        with warnings.catch_warnings():
            # this is just inefficiency in tests, not the tested code
            warnings.filterwarnings("ignore", ".*", sp.SparseEfficiencyWarning)
            a2[0, 1] = 0  # explicitly setting to 0
        self.assertTrue(array_equal(a1, a2))

    def test_csc_scr_equal(self):
        a1 = sp.csc_matrix(([1, 4, 5], ([0, 0, 1], [0, 2, 2])), shape=(2, 3))
        a2 = sp.csr_matrix(([5, 1, 4], ([1, 0, 0], [2, 0, 2])), shape=(2, 3))
        self.assertTrue(array_equal(a1, a2))

        a1 = sp.csc_matrix(([1, 4, 5], ([0, 0, 1], [0, 2, 2])), shape=(2, 3))
        a2 = sp.csr_matrix(([1, 4, 5], ([0, 0, 1], [0, 2, 2])), shape=(2, 3))
        self.assertTrue(array_equal(a1, a2))

    def test_csc_unordered_array_equal(self):
        a1 = sp.csc_matrix(([1, 4, 5], [0, 0, 1], [0, 1, 1, 3]), shape=(2, 3))
        a2 = sp.csc_matrix(([1, 5, 4], [0, 1, 0], [0, 1, 1, 3]), shape=(2, 3))
        self.assertTrue(array_equal(a1, a2))

    def test_wrap_callback(self):
        def func(i):
            return i

        f = wrap_callback(func, start=0, end=0.8)
        self.assertEqual(f(0), 0)
        self.assertEqual(round(f(0.1), 2), 0.08)
        self.assertEqual(f(1), 0.8)

        f = wrap_callback(func, start=0.1, end=0.8)
        self.assertEqual(f(0), 0.1)
        self.assertEqual(f(0.1), 0.17)
        self.assertEqual(f(1), 0.8)


if __name__ == "__main__":
    unittest.main()
