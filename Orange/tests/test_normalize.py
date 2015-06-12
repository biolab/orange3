import unittest

from Orange.data import Table
from Orange.preprocess import Normalize


class TestNormalizer(unittest.TestCase):
    def compare_tables(self, dataNorm, solution):
        for i in range(len(dataNorm)):
            for j in range(len(dataNorm[i])):
                if type(solution[i][j]) == float:
                    self.assertAlmostEqual(dataNorm[i, j], solution[i][j], places=3)
                else:
                    self.assertEqual(dataNorm[i, j], solution[i][j])
        self.assertEqual([attr.name for attr in dataNorm.domain.attributes],
                         ["c1", "c2", "d1", "d2", "n1", "n2", "c3", "d3"])
        self.assertEqual([attr.name for attr in dataNorm.domain.class_vars],
                         ["cl1", "cl2"])

    def test_normalize_default(self):
        data = Table("test5.tab")
        normalizer = Normalize()
        data_norm = normalizer(data)
        solution = [[0., 1.225, 'a', 'a', '?', 'a', 1.225, 'a', 'a', 2],
                    [0., -1.225, 'a', 'b', -1., '?', 0., 'b', 'b', 0],
                    [0., 0., 'a', 'b', 1., 'b', -1.225, 'c', 'c', 1]]
        self.compare_tables(data_norm, solution)

    def test_normalize_transform_by_sd(self):
        data = Table("test5.tab")
        normalizer = Normalize(zero_based=False,
                               norm_type=Normalize.NormalizeBySD,
                               transform_class=False)
        data_norm = normalizer(data)
        solution = [[0., 1.225, 'a', 'a', '?', 'a', 1.225, 'a', 'a', 2],
                    [0., -1.225, 'a', 'b', -1., '?', 0., 'b', 'b', 0],
                    [0., 0., 'a', 'b', 1., 'b', -1.225, 'c', 'c', 1]]
        self.compare_tables(data_norm, solution)

    def test_normalize_transform_class(self):
        data = Table("test5.tab")
        normalizer = Normalize(zero_based=True,
                               norm_type=Normalize.NormalizeBySD,
                               transform_class=True)
        data_norm = normalizer(data)
        solution = [[0., 1.225, 'a', 'a', '?', 'a', 1.225, 'a', 'a', 1.225],
                    [0., -1.225, 'a', 'b', -1., '?', 0., 'b', 'b', -1.225],
                    [0., 0., 'a', 'b', 1., 'b', -1.225, 'c', 'c', 0.]]
        self.compare_tables(data_norm, solution)

    def test_normalize_transform_by_span(self):
        data = Table("test5.tab")
        normalizer = Normalize(zero_based=False,
                               norm_type=Normalize.NormalizeBySpan,
                               transform_class=False)
        data_norm = normalizer(data)
        solution = [[0., 1., 'a', 'a', '?', 'a', 1., 'a', 'a', 2.],
                    [0., -1., 'a', 'b', -1., '?', 0., 'b', 'b', 0.],
                    [0., 0., 'a', 'b', 1., 'b', -1., 'c', 'c', 1.]]
        self.compare_tables(data_norm, solution)

    def test_normalize_transform_by_span_zero(self):
        data = Table("test5.tab")
        normalizer = Normalize(zero_based=True,
                               norm_type=Normalize.NormalizeBySpan,
                               transform_class=False)
        data_norm = normalizer(data)
        solution = [[0., 1., 'a', 'a', '?', 'a', 1., 'a', 'a', 2.],
                    [0., 0., 'a', 'b', 0., '?', 0.5, 'b', 'b', 0.],
                    [0., 0.5, 'a', 'b', 1., 'b', 0., 'c', 'c', 1.]]
        self.compare_tables(data_norm, solution)

    def test_normalize_transform_by_span_class(self):
        data = Table("test5.tab")
        normalizer = Normalize(zero_based=False,
                               norm_type=Normalize.NormalizeBySpan,
                               transform_class=True)
        data_norm = normalizer(data)
        solution = [[0., 1., 'a', 'a', '?', 'a', 1., 'a', 'a', 1.],
                    [0., -1., 'a', 'b', -1., '?', 0., 'b', 'b', -1.],
                    [0., 0., 'a', 'b', 1., 'b', -1., 'c', 'c', 0.]]
        self.compare_tables(data_norm, solution)

    def test_normalize_transform_by_span_zero_class(self):
        data = Table("test5.tab")
        normalizer = Normalize(zero_based=True,
                               norm_type=Normalize.NormalizeBySpan,
                               transform_class=True)
        data_norm = normalizer(data)
        solution = [[0., 1., 'a', 'a', '?', 'a', 1., 'a', 'a', 1.],
                    [0., 0., 'a', 'b', 0., '?', 0.5, 'b', 'b', 0.],
                    [0., 0.5, 'a', 'b', 1., 'b', 0., 'c', 'c', 0.5]]
        self.compare_tables(data_norm, solution)
