# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring

import unittest
from unittest.mock import MagicMock, patch

from Orange.data import Table
from Orange.data.filter import FilterContinuous, FilterDiscrete, Values, HasClass


NIMOCK = MagicMock(side_effect=NotImplementedError())


class FilterTestCase(unittest.TestCase):
    def setUp(self):
        self.iris = Table('iris')

    @patch("Orange.data.Table._filter_values", NIMOCK)
    def test_values(self):
        vs = self.iris.domain.variables
        f1 = FilterContinuous(vs[0], FilterContinuous.Less, 5)
        f2 = FilterContinuous(vs[1], FilterContinuous.Greater, 3)
        f3 = FilterDiscrete(vs[4], [2])
        f12 = Values([f1, f2], conjunction=False, negate=True)
        f123 = Values([f12, f3])
        d12 = f12(self.iris)
        d123 = f123(self.iris)
        self.assertGreater(len(d12), len(d123))
        self.assertTrue((d123.X[:, 0] >= 5).all())
        self.assertTrue((d123.X[:, 1] <= 3).all())
        self.assertTrue((d123.Y == 2).all())
        self.assertEqual(len(d123),
                         (~((self.iris.X[:, 0] < 5) | (self.iris.X[:, 1] > 3)) &
                          (self.iris.Y == 2)).sum())


class TestHasClassFilter(unittest.TestCase):
    def setUp(self):
        self.table = Table('imports-85')
        self.n_missing = 4
        self.assertTrue(self.table.has_missing_class())

    def test_has_class_filter_table(self):
        filter_ = HasClass()
        with_class = filter_(self.table)
        self.assertEqual(len(with_class),
                         len(self.table) - self.n_missing)
        self.assertFalse(with_class.has_missing_class())

        filter_ = HasClass(negate=True)
        without_class = filter_(self.table)
        self.assertEqual(len(without_class), self.n_missing)
        self.assertTrue(without_class.has_missing_class())

    def test_has_class_filter_instance(self):
        class_missing = self.table[9]
        class_present = self.table[0]

        filter_ = HasClass()
        self.assertFalse(filter_(class_missing))
        self.assertTrue(filter_(class_present))

        filter_ = HasClass(negate=True)
        self.assertTrue(filter_(class_missing))
        self.assertFalse(filter_(class_present))

    @patch('Orange.data.Table._filter_has_class', NIMOCK)
    def test_has_class_filter_not_implemented(self):
        self.test_has_class_filter_table()
