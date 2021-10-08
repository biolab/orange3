# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring

import unittest

import numpy as np
import scipy.sparse as sp

from Orange.data import Table, Domain, ContinuousVariable
from Orange.preprocess import Normalize
from Orange.tests import test_filename


class TestNormalizer(unittest.TestCase):
    def compare_tables(self, dataNorm, solution):
        for i in range(len(dataNorm)):
            for j in range(len(dataNorm[i])):
                if type(solution[i][j]) == float:
                    self.assertAlmostEqual(dataNorm[i, j], solution[i][j], places=3)
                else:
                    self.assertEqual(dataNorm[i, j], solution[i][j])
        self.assertEqual([attr.name for attr in dataNorm.domain.attributes],
                         ["c1", "c2", "d1", "d2", "n1", "n2", "c3", "d3", "c4"])
        self.assertEqual([attr.name for attr in dataNorm.domain.class_vars],
                         ["cl1", "cl2"])
    @classmethod
    def setUpClass(cls):
        cls.data = Table(test_filename("datasets/test5.tab"))

    def test_normalize_default(self):
        normalizer = Normalize()
        data_norm = normalizer(self.data)
        solution = [[0., 1.225, 'a', 'a', '?', 'a', 1.225, 'a', '?', 'a', 2],
                    [0., -1.225, 'a', 'b', -1., '?', 0., 'b', '?', 'b', 0],
                    [0., 0., 'a', 'b', 1., 'b', -1.225, 'c', '?', 'c', 1]]
        self.compare_tables(data_norm, solution)

    def test_normalize_transform_by_sd(self):
        normalizer = Normalize(zero_based=False,
                               norm_type=Normalize.NormalizeBySD,
                               transform_class=False)
        data_norm = normalizer(self.data)
        solution = [[0., 1.225, 'a', 'a', '?', 'a', 1.225, 'a', '?', 'a', 2],
                    [0., -1.225, 'a', 'b', -1., '?', 0., 'b', '?', 'b', 0],
                    [0., 0., 'a', 'b', 1., 'b', -1.225, 'c', '?', 'c', 1]]
        self.compare_tables(data_norm, solution)

    def test_normalize_transform_class(self):
        normalizer = Normalize(zero_based=True,
                               norm_type=Normalize.NormalizeBySD,
                               transform_class=True)
        data_norm = normalizer(self.data)
        solution = [[0., 1.225, 'a', 'a', '?', 'a', 1.225, 'a', '?', 'a', 1.225],
                    [0., -1.225, 'a', 'b', -1., '?', 0., 'b', '?', 'b', -1.225],
                    [0., 0., 'a', 'b', 1., 'b', -1.225, 'c', '?', 'c', 0.]]
        self.compare_tables(data_norm, solution)

    def test_normalize_transform_by_span(self):
        normalizer = Normalize(zero_based=False,
                               norm_type=Normalize.NormalizeBySpan,
                               transform_class=False)
        data_norm = normalizer(self.data)
        solution = [[0., 1., 'a', 'a', '?', 'a', 1., 'a', '?', 'a', 2.],
                    [0., -1., 'a', 'b', -1., '?', 0., 'b', '?', 'b', 0.],
                    [0., 0., 'a', 'b', 1., 'b', -1., 'c', '?', 'c', 1.]]
        self.compare_tables(data_norm, solution)

    def test_normalize_transform_by_span_zero(self):
        normalizer = Normalize(zero_based=True,
                               norm_type=Normalize.NormalizeBySpan,
                               transform_class=False)
        data_norm = normalizer(self.data)
        solution = [[0., 1., 'a', 'a', '?', 'a', 1., 'a', '?', 'a', 2.],
                    [0., 0., 'a', 'b', 0., '?', 0.5, 'b', '?', 'b', 0.],
                    [0., 0.5, 'a', 'b', 1., 'b', 0., 'c', '?', 'c', 1.]]
        self.compare_tables(data_norm, solution)

    def test_normalize_transform_by_span_class(self):
        normalizer = Normalize(zero_based=False,
                               norm_type=Normalize.NormalizeBySpan,
                               transform_class=True)
        data_norm = normalizer(self.data)
        solution = [[0., 1., 'a', 'a', '?', 'a', 1., 'a', '?', 'a', 1.],
                    [0., -1., 'a', 'b', -1., '?', 0., 'b', '?', 'b', -1.],
                    [0., 0., 'a', 'b', 1., 'b', -1., 'c', '?', 'c', 0.]]
        self.compare_tables(data_norm, solution)

    def test_normalize_transform_by_span_zero_class(self):
        normalizer = Normalize(zero_based=True,
                               norm_type=Normalize.NormalizeBySpan,
                               transform_class=True)
        data_norm = normalizer(self.data)
        solution = [[0., 1., 'a', 'a', '?', 'a', 1., 'a', '?', 'a', 1.],
                    [0., 0., 'a', 'b', 0., '?', 0.5, 'b', '?', 'b', 0.],
                    [0., 0.5, 'a', 'b', 1., 'b', 0., 'c', '?', 'c', 0.5]]
        self.compare_tables(data_norm, solution)

    def test_normalize_sparse(self):
        domain = Domain([ContinuousVariable(str(i)) for i in range(3)])
        # pylint: disable=bad-whitespace
        X = np.array([
            [0, -1, -2],
            [0,  1,  2],
        ])
        data = Table.from_numpy(domain, X).to_sparse()

        # pylint: disable=bad-whitespace
        solution = sp.csr_matrix(np.array([
            [0, -1, -1],
            [0,  1,  1],
        ]))

        normalizer = Normalize()
        normalized = normalizer(data)
        self.assertEqual((normalized.X != solution).nnz, 0)

        # raise error for non-zero offsets
        with data.unlocked():
            data.X = sp.csr_matrix(np.array([
                [0, 0, 0],
                [0, 1, 3],
                [0, 2, 4],
            ]))
        with self.assertRaises(ValueError):
            normalizer(data)

    def test_skip_normalization(self):
        data = self.data.copy()
        for attr in data.domain.attributes:
            attr.attributes = {'skip-normalization': True}

        normalizer = Normalize()
        normalized = normalizer(data)
        np.testing.assert_array_equal(data.X, normalized.X)

    def test_datetime_normalization(self):
        data = Table(test_filename("datasets/test10.tab"))
        normalizer = Normalize(zero_based=False,
                               norm_type=Normalize.NormalizeBySD,
                               transform_class=False)
        data_norm = normalizer(data)
        solution = [[0., '1995-01-21', 'a', 'a', '?', 'a', 1.225, 'a', '?', 'a', 2],
                    [0., '2003-07-23', 'a', 'b', -1., '?', 0., 'b', '?', 'b', 0],
                    [0., '1967-03-12', 'a', 'b', 1., 'b', -1.225, 'c', '?', 'c', 1]]
        self.compare_tables(data_norm, solution)

    def test_retain_vars_attributes(self):
        data = Table("iris")
        attributes = {"foo": "foo", "baz": 1}
        data.domain.attributes[0].attributes = attributes
        self.assertDictEqual(
            Normalize(norm_type=Normalize.NormalizeBySD)(
                data).domain.attributes[0].attributes, attributes)
        self.assertDictEqual(
            Normalize(norm_type=Normalize.NormalizeBySpan)(
                data).domain.attributes[0].attributes, attributes)

    def test_number_of_decimals(self):
        foo = ContinuousVariable("Foo", number_of_decimals=0)
        data = Table.from_list(Domain((foo,)), [[1], [2], [3]])

        normalized = Normalize()(data)
        norm_foo: ContinuousVariable = normalized.domain.attributes[0]

        self.assertGreater(norm_foo.number_of_decimals, 0)

        for val1, val2 in zip(normalized[:, "Foo"],
                              ["-1.225", "0.0", "1.225"]):
            self.assertEqual(str(val1[0]), val2)


if __name__ == "__main__":
    unittest.main()
