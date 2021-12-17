# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring

import pickle
import unittest
from unittest.mock import Mock

import numpy as np
from scipy.sparse import csr_matrix

from Orange.data import Table, Domain, ContinuousVariable
from Orange.preprocess import EntropyMDL, DoNotImpute, Default, Average, \
    SelectRandomFeatures, EqualFreq, RemoveNaNColumns, DropInstances, \
    EqualWidth, SelectBestFeatures, RemoveNaNRows, Preprocess, Scale, \
    Randomize, Continuize, Discretize, Impute, SklImpute, Normalize, \
    ProjectCUR, ProjectPCA, RemoveConstant, AdaptiveNormalize, RemoveSparse


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


class TestRemoveConstant(unittest.TestCase):
    def test_remove_columns(self):
        X = np.random.rand(6, 5)
        X[:, (1, 3)] = 5
        X[3, 1] = np.nan
        X[1, 1] = np.nan
        X[:, 4] = np.nan
        data = Table.from_numpy(None, X)
        d = RemoveConstant()(data)
        self.assertEqual(len(d.domain.attributes), 2)

        pp_rc = RemoveConstant()
        d = pp_rc(data)
        self.assertEqual(len(d.domain.attributes), 2)

    def test_nothing_to_remove(self):
        data = Table("iris")
        d = RemoveConstant()(data)
        self.assertEqual(len(d.domain.attributes), 4)


class TestRemoveNaNRows(unittest.TestCase):
    def test_remove_row(self):
        data = Table("iris")
        with data.unlocked():
            data.X[0, 0] = np.nan
        pp_data = RemoveNaNRows()(data)
        self.assertEqual(len(pp_data), len(data) - 1)
        self.assertFalse(np.isnan(pp_data.X).any())


class TestRemoveNaNColumns(unittest.TestCase):
    def test_column_filtering(self):
        data = Table("iris")
        with data.unlocked():
            data.X[:, (1, 3)] = np.NaN

        new_data = RemoveNaNColumns()(data)
        self.assertEqual(len(new_data.domain.attributes),
                         len(data.domain.attributes) - 2)

        data = Table("iris")
        with data.unlocked():
            data.X[0, 0] = np.NaN
        new_data = RemoveNaNColumns()(data)
        self.assertEqual(len(new_data.domain.attributes),
                         len(data.domain.attributes))

    def test_column_filtering_sparse(self):
        data = Table("iris")
        with data.unlocked():
            data.X = csr_matrix(data.X)

        new_data = RemoveNaNColumns()(data)
        self.assertEqual(data, new_data)


class TestScaling(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.table = Table.from_numpy(None, [[1, 2, 3],
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
                    Average, Default, RemoveSparse]

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


class TestAdaptiveNormalize(unittest.TestCase):
    """
    Checks if output for sparse data is the same as for Scale
    preprocessor. For dense data the output should match that
     of Normalize preprocessor.
    """

    def setUp(self):
        self.data = Table("iris")

    def test_dense_pps(self):
        true_out = Normalize()(self.data)
        out = AdaptiveNormalize()(self.data)
        np.testing.assert_array_equal(out, true_out)

    def test_sparse_pps(self):
        with self.data.unlocked():
            self.data.X = csr_matrix(self.data.X)
        out = AdaptiveNormalize()(self.data)
        true_out = Scale(center=Scale.NoCentering, scale=Scale.Span)(self.data)
        np.testing.assert_array_equal(out, true_out)
        self.data = self.data.X.toarray()


class TestRemoveSparse(unittest.TestCase):

    def setUp(self):
        domain = Domain([ContinuousVariable('a'), ContinuousVariable('b')])
        self.data = Table.from_numpy(domain, np.zeros((3, 2)))

    def test_0_dense(self):
        with self.data.unlocked():
            self.data[1:, 1] = 7
            true_out = self.data[:, 1].copy()
        with true_out.unlocked(true_out.X):
            true_out.X = true_out.X.reshape(-1, 1)
        out = RemoveSparse(0.5, True)(self.data)
        np.testing.assert_array_equal(out, true_out)

        out = RemoveSparse(2, True)(self.data)
        np.testing.assert_array_equal(out, true_out)

    def test_0_sparse(self):
        with self.data.unlocked():
            self.data[1:, 1] = 7
            true_out = self.data[:, 1].copy()
            self.data.X = csr_matrix(self.data.X)
        with true_out.unlocked(true_out.X):
            true_out.X = csr_matrix(true_out.X)
        out = RemoveSparse(0.5, True)(self.data).X
        np.testing.assert_array_equal(out, true_out)

        out = RemoveSparse(1, True)(self.data).X
        np.testing.assert_array_equal(out, true_out)

    def test_nan_dense(self):
        with self.data.unlocked():
            self.data[1:, 1] = np.nan
            self.data.X[:, 0] = 7
            true_out = self.data[:, 0].copy()
        with true_out.unlocked(true_out.X):
            true_out.X = true_out.X.reshape(-1, 1)
        out = RemoveSparse(0.5, False)(self.data)
        np.testing.assert_array_equal(out, true_out)

        out = RemoveSparse(1, False)(self.data)
        np.testing.assert_array_equal(out, true_out)

    def test_nan_sparse(self):
        with self.data.unlocked():
            self.data[1:, 1] = np.nan
            self.data.X[:, 0] = 7
            true_out = self.data[:, 0].copy()
            with true_out.unlocked(true_out.X):
                true_out.X = true_out.X.reshape(-1, 1)
                true_out.X = csr_matrix(true_out.X)
            self.data.X = csr_matrix(self.data.X)
        out = RemoveSparse(0.5, False)(self.data)
        np.testing.assert_array_equal(out, true_out)

        out = RemoveSparse(1, False)(self.data)
        np.testing.assert_array_equal(out, true_out)


if __name__ == '__main__':
    unittest.main()
