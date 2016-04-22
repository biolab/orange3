# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring

import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from Orange.data import Table, Domain, ContinuousVariable
from Orange.data.filter import \
    FilterContinuous, FilterDiscrete, Values, HasClass

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


class TestFilterContinuous(unittest.TestCase):
    def setUp(self):
        self.domain = Domain([ContinuousVariable(x) for x in "abcd"])
        self.inst = Table(self.domain, np.array([[0.1, 0.2, 0.3, np.nan]]))[0]

    def test_min(self):
        flt = FilterContinuous(1, FilterContinuous.Between, 1, 2)
        self.assertEqual(flt.min, 1)
        self.assertEqual(flt.max, 2)
        self.assertEqual(flt.ref, 1)

        flt.ref = 0
        self.assertEqual(flt.min, 0)

        flt.min = -1
        self.assertEqual(flt.ref, -1)

        self.assertRaises(
            TypeError,
            FilterContinuous, 1, FilterContinuous.Equal, 0, c=12)
        self.assertRaises(
            TypeError,
            FilterContinuous, 1, FilterContinuous.Equal, 0, min=5, c=12)

        flt = FilterContinuous(1, FilterContinuous.Between, min=1, max=2)
        self.assertEqual(flt.ref, 1)

    def test_operator(self):
        inst = self.inst
        flt = FilterContinuous
        self.assertTrue(flt(1, flt.Equal, 0.2)(inst))
        self.assertFalse(flt(1, flt.Equal, 0.3)(inst))

        self.assertTrue(flt(1, flt.NotEqual, 0.3)(inst))
        self.assertFalse(flt(1, flt.NotEqual, 0.2)(inst))

        self.assertTrue(flt(1, flt.Less, 0.3)(inst))
        self.assertFalse(flt(1, flt.Less, 0.2)(inst))

        self.assertTrue(flt(1, flt.LessEqual, 0.3)(inst))
        self.assertTrue(flt(1, flt.LessEqual, 0.2)(inst))
        self.assertFalse(flt(1, flt.LessEqual, 0.1)(inst))

        self.assertTrue(flt(1, flt.Greater, 0.1)(inst))
        self.assertFalse(flt(1, flt.Greater, 0.2)(inst))

        self.assertTrue(flt(1, flt.GreaterEqual, 0.1)(inst))
        self.assertTrue(flt(1, flt.GreaterEqual, 0.2)(inst))
        self.assertFalse(flt(1, flt.GreaterEqual, 0.3)(inst))

        self.assertTrue(flt(1, flt.Between, 0.05, 0.4)(inst))
        self.assertTrue(flt(1, flt.Between, 0.2, 0.4)(inst))
        self.assertTrue(flt(1, flt.Between, 0.05, 0.2)(inst))
        self.assertFalse(flt(1, flt.Between, 0.3, 0.4)(inst))

        self.assertFalse(flt(1, flt.Outside, 0.05, 0.4)(inst))
        self.assertFalse(flt(1, flt.Outside, 0.2, 0.4)(inst))
        self.assertFalse(flt(1, flt.Outside, 0.05, 0.2)(inst))
        self.assertTrue(flt(1, flt.Outside, 0.3, 0.4)(inst))

        self.assertTrue(flt(1, flt.IsDefined)(inst))
        self.assertFalse(flt(3, flt.IsDefined)(inst))

        self.assertRaises(ValueError, flt(1, -1, 1), inst)

    def test_position(self):
        inst = self.inst
        flt = FilterContinuous
        self.assertFalse(flt(0, flt.Equal, 0.2)(inst))
        self.assertTrue(flt(1, flt.Equal, 0.2)(inst))
        self.assertFalse(flt(2, flt.Equal, 0.2)(inst))
        self.assertFalse(flt(3, flt.Equal, 0.2)(inst))

        self.assertFalse(flt("a", flt.Equal, 0.2)(inst))
        self.assertTrue(flt("b", flt.Equal, 0.2)(inst))
        self.assertFalse(flt("c", flt.Equal, 0.2)(inst))
        self.assertFalse(flt("d", flt.Equal, 0.2)(inst))

        a, b, c, d = self.domain.attributes
        self.assertFalse(flt(a, flt.Equal, 0.2)(inst))
        self.assertTrue(flt(b, flt.Equal, 0.2)(inst))
        self.assertFalse(flt(c, flt.Equal, 0.2)(inst))
        self.assertFalse(flt(d, flt.Equal, 0.2)(inst))

    def test_nan(self):
        inst = self.inst
        flt = FilterContinuous

        self.assertFalse(flt(3, flt.Equal, 0.3)(inst))
        self.assertFalse(flt(3, flt.NotEqual, 0.3)(inst))
        self.assertFalse(flt(3, flt.Less, 0.2)(inst))
        self.assertFalse(flt(3, flt.LessEqual, 0.1)(inst))
        self.assertFalse(flt(3, flt.Greater, 0.2)(inst))
        self.assertFalse(flt(3, flt.GreaterEqual, 0.1)(inst))
        self.assertFalse(flt(3, flt.Between, 0.05, 0.4)(inst))
        self.assertFalse(flt(3, flt.Outside, 0.05, 0.4)(inst))

        self.assertTrue(flt(3, flt.Equal, np.nan)(inst))
        self.assertFalse(flt(3, flt.NotEqual, np.nan)(inst))

    def test_str(self):
        flt = FilterContinuous(1, FilterContinuous.Equal, 1)

        self.assertEqual(str(flt), "feature(1) = 1")

        flt = FilterContinuous("foo", FilterContinuous.Equal, 1)
        self.assertEqual(str(flt), "foo = 1")

        flt = FilterContinuous(self.domain[0], FilterContinuous.Equal, 1, 2)
        self.assertEqual(str(flt), "a = 1")

        flt.oper = flt.NotEqual
        self.assertEqual(str(flt), "a ≠ 1")

        flt.oper = flt.Less
        self.assertEqual(str(flt), "a < 1")

        flt.oper = flt.LessEqual
        self.assertEqual(str(flt), "a ≤ 1")

        flt.oper = flt.Greater
        self.assertEqual(str(flt), "a > 1")

        flt.oper = flt.GreaterEqual
        self.assertEqual(str(flt), "a ≥ 1")

        flt.oper = flt.Between
        self.assertEqual(str(flt), "1 ≤ a ≤ 2")

        flt.oper = flt.Outside
        self.assertEqual(str(flt), "not 1 ≤ a ≤ 2")

        flt.oper = flt.IsDefined
        self.assertEqual(str(flt), "a is defined")

        flt.oper = -1
        self.assertEqual(str(flt), "invalid operator")
