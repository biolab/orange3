# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring

import unittest
from unittest.mock import MagicMock, patch

from Orange.data import Table
from Orange.data.filter import FilterContinuous, FilterDiscrete, Values, HasClass, \
    SameValue

import itertools

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


class TestSameValueFilter(unittest.TestCase):
    def setUp(self):
        self.table = Table('zoo')

        self.attr_disc  = self.table.domain["type"]
        self.attr_cont  = self.table.domain["legs"]
        self.attr_meta  = self.table.domain["name"]

        self.value_cont = 4
        self.value_disc = self.attr_disc.to_val("mammal")
        self.value_meta = self.attr_meta.to_val("girl")

    def test_same_value_filter_table(self):

        test_pairs = ((self.attr_cont,  4, self.value_cont),
                      (self.attr_disc, "mammal", self.value_disc),
                      (self.attr_meta, "girl", self.value_meta),)

        for pos, val, r in test_pairs:
                filter_ = SameValue(pos, val)(self.table)
                self.assertTrue(all(inst[pos] == r for inst in filter_))

                filter_n = SameValue(pos, val, negate=True)(self.table)
                self.assertTrue(all(inst[pos] != r for inst in filter_n))

                self.assertEqual(len(filter_) + len(filter_n), len(self.table))


        for t1, t2 in itertools.combinations(test_pairs, 2):
            pos1, val1, r1 = t1
            pos2, val2, r2 = t2

            filter_1 = SameValue(pos1, val1)(self.table)
            filter_2 = SameValue(pos2, val2)(self.table)

            filter_12 = SameValue(pos2, val2)(SameValue(pos1, val1)(self.table))
            filter_21 = SameValue(pos1, val1)(SameValue(pos2, val2)(self.table))

            self.assertEqual(len(filter_21), len(filter_12))

            self.assertTrue(len(filter_1) >= len(filter_12))
            self.assertTrue(len(filter_2) >= len(filter_12))

            self.assertTrue(all(inst[pos1] == r1 and inst[pos2] == r2
                                    and inst in filter_21
                                    for inst in filter_12))
            self.assertTrue(all(inst[pos1] == r1 and inst[pos2] == r2
                                and inst in filter_12
                                for inst in filter_21))

    def test_same_value_filter_instance(self):
        inst = self.table[0]

        filter_ = SameValue(self.attr_disc, self.value_disc)(inst)
        self.assertEqual(filter_, inst[self.attr_disc] == self.value_disc)

        filter_n = SameValue(self.attr_disc, self.value_disc, negate=True)(inst)
        self.assertEqual(filter_n, inst[self.attr_disc] != self.value_disc)

    @patch('Orange.data.Table._filter_same_value', NIMOCK)
    def test_has_class_filter_not_implemented(self):
        self.test_same_value_filter_table()
