import os
import unittest
import warnings

import numpy as np
import scipy.sparse as sp

from Orange.util import export_globals, flatten, deprecated, try_, deepgetattr, \
    OrangeDeprecationWarning
from Orange.data import Table
from Orange.data.util import vstack, hstack
from Orange.statistics.util import stats

SOMETHING = 0xf00babe


class TestUtil(unittest.TestCase):
    def test_export_globals(self):
        self.assertEqual(sorted(export_globals(globals(), __name__)),
                         ['SOMETHING', 'TestUtil'])

    def test_flatten(self):
        self.assertEqual(list(flatten([[1, 2], [3]])), [1, 2, 3])

    def test_deprecated(self):
        @deprecated
        def identity(x): return x

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
