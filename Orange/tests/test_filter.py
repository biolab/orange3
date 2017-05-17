# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring

import unittest
from unittest.mock import MagicMock, patch
import itertools

import numpy as np

from Orange.data import Table, Domain, ContinuousVariable, DiscreteVariable, \
    StringVariable
from Orange.data.filter import \
    FilterContinuous, FilterDiscrete, FilterString, Values, HasClass, \
    IsDefined, SameValue, Random, ValueFilter, FilterStringList, FilterRegex

NIMOCK = MagicMock(side_effect=NotImplementedError())


class TestFilterValues(unittest.TestCase):
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


class TestIsDefinedFilter(unittest.TestCase):
    def setUp(self):
        self.table = Table('imports-85')
        self.n_missing = 46
        self.assertTrue(self.table.has_missing())

    def test_is_defined_filter_table(self):
        filter_ = IsDefined()
        without_missing = filter_(self.table)
        self.assertEqual(len(without_missing),
                         len(self.table) - self.n_missing)
        self.assertFalse(without_missing.has_missing())

        filter_ = IsDefined(negate=True)
        just_missing = filter_(self.table)
        self.assertEqual(len(just_missing), self.n_missing)
        self.assertTrue(just_missing.has_missing())

    def test_is_defined_filter_instance(self):
        instance_with_missing = self.table[0]
        instance_without_missing = self.table[3]

        filter_ = IsDefined()
        self.assertFalse(filter_(instance_with_missing))
        self.assertTrue(filter_(instance_without_missing))

        filter_ = IsDefined(negate=True)
        self.assertTrue(filter_(instance_with_missing))
        self.assertFalse(filter_(instance_without_missing))

    @patch('Orange.data.Table._filter_is_defined', NIMOCK)
    def test_is_defined_filter_not_implemented(self):
        self.test_is_defined_filter_table()


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

    def test_eq(self):
        flt1 = FilterContinuous(1, FilterContinuous.Between, 1, 2)
        flt2 = FilterContinuous(1, FilterContinuous.Between, 1, 2)
        flt3 = FilterContinuous(1, FilterContinuous.Between, 1, 3)
        self.assertEqual(flt1, flt2)
        self.assertNotEqual(flt1, flt3)
        self.assertEqual(flt1.__dict__ == flt2.__dict__, flt1 == flt2)
        self.assertEqual(flt1.__dict__ == flt3.__dict__, flt1 == flt3)


class TestFilterDiscrete(unittest.TestCase):
    def test_eq(self):
        flt1 = FilterDiscrete(1, None)
        flt2 = FilterDiscrete(1, None)
        flt3 = FilterDiscrete(2, None)
        self.assertEqual(flt1, flt2)
        self.assertEqual(flt1.__dict__ == flt2.__dict__, flt1 == flt2)
        self.assertNotEqual(flt1, flt3)
        self.assertEqual(flt1.__dict__ == flt3.__dict__, flt1 == flt3)


class TestFilterString(unittest.TestCase):

    def setUp(self):
        self.data = Table("zoo")
        self.inst = self.data[0]  # aardvark

    def test_case_sensitive(self):
        flt = FilterString("name", FilterString.Equal, "Aardvark", case_sensitive=True)
        self.assertFalse(flt(self.inst))
        flt = FilterString("name", FilterString.Equal, "Aardvark", case_sensitive=False)
        self.assertTrue(flt(self.inst))

    def test_operators(self):
        flt = FilterString("name", FilterString.Equal, "aardvark")
        self.assertTrue(flt(self.inst))
        flt = FilterString("name", FilterString.Equal, "bass")
        self.assertFalse(flt(self.inst))

        flt = FilterString("name", FilterString.NotEqual, "bass")
        self.assertTrue(flt(self.inst))
        flt = FilterString("name", FilterString.NotEqual, "aardvark")
        self.assertFalse(flt(self.inst))

        flt = FilterString("name", FilterString.Less, "bass")
        self.assertTrue(flt(self.inst))
        flt = FilterString("name", FilterString.Less, "aa")
        self.assertFalse(flt(self.inst))

        flt = FilterString("name", FilterString.LessEqual, "bass")
        self.assertTrue(flt(self.inst))
        flt = FilterString("name", FilterString.LessEqual, "aardvark")
        self.assertTrue(flt(self.inst))
        flt = FilterString("name", FilterString.LessEqual, "aa")
        self.assertFalse(flt(self.inst))

        flt = FilterString("name", FilterString.Greater, "aa")
        self.assertTrue(flt(self.inst))
        flt = FilterString("name", FilterString.Greater, "aardvark")
        self.assertFalse(flt(self.inst))

        flt = FilterString("name", FilterString.GreaterEqual, "aa")
        self.assertTrue(flt(self.inst))
        flt = FilterString("name", FilterString.GreaterEqual, "aardvark")
        self.assertTrue(flt(self.inst))
        flt = FilterString("name", FilterString.GreaterEqual, "bass")
        self.assertFalse(flt(self.inst))

        flt = FilterString("name", FilterString.Between, "aa", "aardvark")
        self.assertTrue(flt(self.inst))
        flt = FilterString("name", FilterString.Between, "a", "aa")
        self.assertFalse(flt(self.inst))

        flt = FilterString("name", FilterString.Outside, "aaz", "bass")
        self.assertTrue(flt(self.inst))
        flt = FilterString("name", FilterString.Outside, "aardvark", "bass")
        self.assertFalse(flt(self.inst))

        flt = FilterString("name", FilterString.Contains, "ard")
        self.assertTrue(flt(self.inst))
        flt = FilterString("name", FilterString.Contains, "ra")
        self.assertFalse(flt(self.inst))

        flt = FilterString("name", FilterString.StartsWith, "aar")
        self.assertTrue(flt(self.inst))
        flt = FilterString("name", FilterString.StartsWith, "ard")
        self.assertFalse(flt(self.inst))

        flt = FilterString("name", FilterString.EndsWith, "aardvark")
        self.assertTrue(flt(self.inst))
        flt = FilterString("name", FilterString.EndsWith, "aard")
        self.assertFalse(flt(self.inst))

        flt = FilterString("name", FilterString.IsDefined)
        self.assertTrue(flt(self.inst))
        for s in ["?", "nan"]:
            self.inst["name"] = s
            flt = FilterString("name", FilterString.IsDefined)
            self.assertTrue(flt(self.inst))
        self.inst["name"] = ""
        flt = FilterString("name", FilterString.IsDefined)
        self.assertFalse(flt(self.inst))


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

        test_pairs = ((self.attr_cont, 4, self.value_cont),
                      (self.attr_disc, "mammal", self.value_disc),
                      (self.attr_meta, "girl", self.value_meta),)

        for var_index, value, num_value in test_pairs:
            filter_ = SameValue(var_index, value)(self.table)
            self.assertTrue(all(inst[var_index] == num_value for inst in filter_))

            filter_inverse = SameValue(var_index, value, negate=True)(self.table)
            self.assertTrue(all(inst[var_index] != num_value for inst in filter_inverse))

            self.assertEqual(len(filter_) + len(filter_inverse), len(self.table))


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

            self.assertTrue(all(inst[pos1] == r1 and
                                inst[pos2] == r2 and
                                inst in filter_21
                                for inst in filter_12))
            self.assertTrue(all(inst[pos1] == r1 and
                                inst[pos2] == r2 and
                                inst in filter_12
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


class TestFilterReprs(unittest.TestCase):
    def setUp(self):
        self.table = Table('zoo')
        self.attr_disc  = self.table.domain["type"]
        self.value_disc = self.attr_disc.to_val("mammal")
        self.vs = self.table.domain.variables

        self.table2 = Table("zoo")
        self.inst = self.table2[0]  # aardvark

    def test_reprs(self):
        flid = IsDefined(negate=True)
        flhc = HasClass()
        flr = Random()
        fld = FilterDiscrete(self.attr_disc, None)
        flsv = SameValue(self.attr_disc, self.value_disc, negate=True)
        flc = FilterContinuous(self.vs[0], FilterContinuous.Less, 5)
        flc2 = FilterContinuous(self.vs[1], FilterContinuous.Greater, 3)
        flv = Values([flc, flc2], conjunction=False, negate=True)
        flvf = ValueFilter(self.attr_disc)
        fls = FilterString("name", FilterString.Equal, "Aardvark", case_sensitive=False)
        flsl = FilterStringList("name", ["Aardvark"], case_sensitive=False)
        flrx = FilterRegex("name", "^c...$")

        filters = [flid, flhc, flr, fld, flsv, flc, flv, flvf, fls, flsl, flrx]

        for f in filters:
            repr_str = repr(f)
            print(repr_str)
            new_f = eval(repr_str)
            self.assertEqual(repr(new_f), repr_str)
