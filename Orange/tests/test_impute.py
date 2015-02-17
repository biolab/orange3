import unittest

import numpy as np

from Orange import preprocess
from Orange import data
from Orange.data import Unknown


class TestReplaceUnknowns(unittest.TestCase):
    def test_replacement(self):
        a = np.arange(10, dtype=float)
        a[1] = a[5] = Unknown
        ia = preprocess.ReplaceUnknowns(None).transform(a)
        np.testing.assert_equal(ia, [0, 0, 2, 3, 4, 0, 6, 7, 8, 9])

        a[1] = a[5] = Unknown
        ia = preprocess.ReplaceUnknowns(None, value=42).transform(a)
        np.testing.assert_equal(ia, [0, 42, 2, 3, 4, 42, 6, 7, 8, 9])


class TestAverage(unittest.TestCase):
    def test_replacement(self):
        s = [0] * 50 + [1] * 50
        c1 = np.array(s).reshape((100, 1))
        s = [0] * 5 + [1] * 5 + [2] * 90
        c2 = np.array(s).reshape((100, 1))
        x = np.hstack([c1, c2])
        domain = data.Domain([data.ContinuousVariable("a"),
                              data.DiscreteVariable("b", values="ABC")],
                             data.ContinuousVariable("c"),)
        table = data.Table(domain, x, c1)
        var1 = preprocess.Average()(table, 0)
        self.assertIsInstance(var1.compute_value, preprocess.ReplaceUnknowns)
        self.assertEqual(var1.compute_value.value, 0.5)
        var2 = preprocess.Average()(table, 1)
        self.assertIsInstance(var2.compute_value, preprocess.ReplaceUnknowns)
        self.assertEqual(var2.compute_value.value, 2)


class TestImputer(unittest.TestCase):
    def test_imputer(self):
        auto = data.Table('auto-mpg')
        auto2 = preprocess.Impute(auto)
        self.assertFalse(np.isnan(auto2.X).any())
