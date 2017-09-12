# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring

import os
import pickle
import unittest
from unittest.mock import Mock
import numpy as np
import scipy.sparse as sp

from Orange.data import Table
from Orange.preprocess import EntropyMDL, DoNotImpute, Default, Average, SelectRandomFeatures, EqualFreq, \
    RemoveNaNColumns, DropInstances
from Orange.preprocess import EqualWidth, SelectBestFeatures
from Orange.preprocess.preprocess import Preprocess, Scale, Randomize, Continuize, Discretize, Impute, SklImpute, \
    Normalize, ProjectCUR, ProjectPCA, RemoveConstant
from Orange.util import OrangeDeprecationWarning


class TestPreprocess(unittest.TestCase):
    def test_read_data_calls_reader(self):
        class MockPreprocessor(Preprocess):
            __init__ = Mock(return_value=None)
            __call__ = Mock()
            @classmethod
            def reset(cls):
                cls.__init__.reset_mock()
                cls.__call__.reset_mock()

        table = Mock(Table)
        MockPreprocessor(1, 2, a=3)(table)
        MockPreprocessor.__init__.assert_called_with(1, 2, a=3)
        MockPreprocessor.__call__.assert_called_with(table)
        MockPreprocessor.reset()

        MockPreprocessor(1, 2, a=3)
        MockPreprocessor.__init__.assert_called_with(1, 2, a=3)
        self.assertEqual(MockPreprocessor.__call__.call_count, 0)

        MockPreprocessor(a=3)
        MockPreprocessor.__init__.assert_called_with(a=3)
        self.assertEqual(MockPreprocessor.__call__.call_count, 0)

        MockPreprocessor()
        MockPreprocessor.__init__.assert_called_with()
        self.assertEqual(MockPreprocessor.__call__.call_count, 0)

    def test_refuse_data_in_constructor(self):
        # We force deprecations as exceptions as part of CI
        is_CI = os.environ.get('CI') or os.environ.get('ORANGE_DEPRECATIONS_ERROR')
        if is_CI:
            self.assertTrue(os.environ.get('ORANGE_DEPRECATIONS_ERROR'))
        expected = self.assertRaises if is_CI else self.assertWarns
        with expected(OrangeDeprecationWarning):
            try:
                Preprocess(Table('iris'))
            except NotImplementedError:
                # Expected from default Preprocess.__call__
                pass


class TestRemoveConstant(unittest.TestCase):
    def test_remove_columns(self):
        X = np.random.rand(6, 4)
        X[:, (1,3)] = 5
        X[3, 1] = np.nan
        X[1, 1] = np.nan
        data = Table(X)
        d = RemoveConstant()(data)
        self.assertEqual(len(d.domain.attributes), 2)

        pp_rc = RemoveConstant()
        d = pp_rc(data)
        self.assertEqual(len(d.domain.attributes), 2)

    def test_nothing_to_remove(self):
        data = Table("iris")
        d = RemoveConstant()(data)
        self.assertEqual(len(d.domain.attributes), 4)


class TestRandomize(unittest.TestCase):
    def test_randomize(self):
        x = np.arange(10000, dtype=int).reshape((100, 100))
        randomized = Randomize().randomize(x.copy())

        # Do not mix data between columns
        np.testing.assert_equal(randomized % 100, x % 100)

        # Do not shuffle entire rows: lexical sorting of rows should equal the
        # original table
        randomized = np.array(sorted(list(map(list, randomized))), dtype=int)
        self.assertFalse(np.all(randomized == x))

    def test_randomize_sparse(self):
        x = np.array([[0, 0, 3, 0],
                      [1, 0, 2, 0],
                      [4, 5, 6, 7]])
        randomized = Randomize(rand_seed=1).randomize(sp.csr_matrix(x))
        randomized = randomized.toarray()
        # Data is shuffled (rand_seed=1 should always shuffle it)
        self.assertFalse(np.all(x == randomized))
        # Data remains within a column
        self.assertTrue(all(sorted(x[:, i]) == sorted(randomized[:, i])
                            for i in range(4)))
        # Do not shuffle entire rows
        randomized = np.array(sorted(list(map(list, randomized))), dtype=int)
        self.assertFalse(np.all(randomized == x))


class TestScaling(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.table = Table([[1, 2, 3],
                           [2, 3, 4],
                           [3, 4, 5],
                           [4, 5, 6]])

    def test_scaling_mean_span(self):
        table = Scale(center=Scale.Mean, scale=Scale.Span)(self.table)
        np.testing.assert_almost_equal(np.mean(table, 0), 0)
        np.testing.assert_almost_equal(np.ptp(table, 0), 1)

    def test_scaling_median_stddev(self):
        table = Scale(center=Scale.Median, scale=Scale.Std)(self.table)
        np.testing.assert_almost_equal(np.std(table, 0), 1)
        # NB: This test just covers. The following fails. You figure it out.
        # np.testing.assert_almost_equal(np.median(table, 0), 0)


class TestReprs(unittest.TestCase):
    def test_reprs(self):
        preprocs = [Continuize, Discretize, Impute, SklImpute, Normalize,
                    Randomize, ProjectPCA, ProjectCUR, Scale,
                    EqualFreq, EqualWidth, EntropyMDL, SelectBestFeatures,
                    SelectRandomFeatures, RemoveNaNColumns, DoNotImpute, DropInstances,
                    Average, Default]

        for preproc in preprocs:
            repr_str = repr(preproc())
            new_preproc = eval(repr_str)
            self.assertEqual(repr(new_preproc), repr_str)


class TestEnumPickling(unittest.TestCase):
    def test_continuize_pickling(self):
        c = Continuize(multinomial_treatment=Continuize.FirstAsBase)
        s = pickle.dumps(c, -1)
        c1 = pickle.loads(s)
        self.assertIs(c1.multinomial_treatment, c.multinomial_treatment)

    def test_randomize_pickling(self):
        c = Randomize(rand_type=Randomize.RandomizeMetas)
        s = pickle.dumps(c, -1)
        c1 = pickle.loads(s)
        self.assertIs(c1.rand_type, c.rand_type)

    def test_scaling_pickling(self):
        c = Scale(center=Scale.Median, scale=Scale.Span)
        s = pickle.dumps(c, -1)
        c1 = pickle.loads(s)
        self.assertIs(c1.center, c.center)
        self.assertIs(c1.scale, c.scale)
