# pylint: disable=missing-docstring

import unittest
import numpy as np

from Orange.data import Table
from Orange.widgets.data.owselectrows import Filter


class TestSelectRows(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.table = Table()
        cls.table["a"] = [1, 2, 3, 4, 5]
        cls.table["b"] = ["c", "b", "c", "d", "c"]
        cls.table["c"] = [3, 2, 1, 2, 3]
        cls.table["d"] = [np.nan, 2, 3, 4, np.nan]
        cls.table["e"] = ["tralala", "hopsala", "a", "trla", ""]

    def test_filters_equals(self):
        self.assertEqual(2, len(self.table[Filter.Equals(self.table.c, 2)]))
        self.assertEqual(3, len(self.table[Filter.Equals(self.table.b, "c")]))

    def test_filter_isnot(self):
        self.assertEqual(3, len(self.table[Filter.IsNot(self.table.c, 2)]))
        self.assertEqual(2, len(self.table[Filter.IsNot(self.table.b, "c")]))

    def test_filter_isbelow(self):
        self.assertEqual(5, len(self.table[Filter.IsBelow(self.table.a, 100)]))

    def test_filter_isatmost(self):
        self.assertEqual(3, len(self.table[Filter.IsAtMost(self.table.c, 2)]))

    def test_filter_isgreaterthan(self):
        self.assertEqual(1, len(self.table[Filter.IsGreaterThan(self.table.a, 4.99)]))

    def test_filter_isatleast(self):
        self.assertEqual(2, len(self.table[Filter.IsAtLeast(self.table.c, 3)]))

    def test_filter_isbetween(self):
        self.assertEqual(3, len(self.table[Filter.IsBetween(self.table.a, 2, 4)]))

    def test_filter_isoutside(self):
        self.assertEqual(2, len(self.table[Filter.IsOutside(self.table.a, 2, 4)]))

    def test_filter_isdefined(self):
        self.assertEqual(3, len(self.table[Filter.IsDefined(self.table.d)]))

    def test_filter_is(self):
        self.assertEqual(3, len(self.table[Filter.Is(self.table.b, "c")]))

    def test_filter_isoneof(self):
        self.assertEqual(2, len(self.table[Filter.IsOneOf(self.table.b, ("b", "d"))]))

    def test_filter_isbefore(self):
        self.assertEqual(1, len(self.table[Filter.IsBefore(self.table.b, "c")]))

    def test_filter_isequalorbefore(self):
        self.assertEqual(4, len(self.table[Filter.IsEqualOrBefore(self.table.b, "c")]))

    def test_filter_isafter(self):
        self.assertEqual(1, len(self.table[Filter.IsAfter(self.table.b, "c")]))

    def test_filter_isequalorafter(self):
        self.assertEqual(4, len(self.table[Filter.IsEqualOrAfter(self.table.b, "c")]))

    def test_filter_contains(self):
        self.assertEqual(3, len(self.table[Filter.Contains(self.table.e, "la")]))

    def test_filter_beginswith(self):
        self.assertEqual(2, len(self.table[Filter.BeginsWith(self.table.e, "tr")]))

    def test_filter_endswith(self):
        self.assertEqual(3, len(self.table[Filter.EndsWith(self.table.e, "la")]))

    def test_filter_chain(self):
        f1 = Filter.EndsWith(self.table.e, "la")
        f2 = Filter.IsDefined(self.table.c)
        f3 = Filter.IsAtMost(self.table.c, 2)
        ff = f1 & f2 & f3
        self.assertEqual(2, len(self.table[ff]))
