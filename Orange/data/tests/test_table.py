import unittest

import numpy as np

from Orange.data import (
    ContinuousVariable, DiscreteVariable, StringVariable,
    Domain, Table, IsDefined, FilterContinuous, Values, FilterString,
    FilterDiscrete, FilterStringList, FilterRegex)


class TestTableFilters(unittest.TestCase):
    def setUp(self):
        self.domain = Domain(
            [ContinuousVariable("c1"),
             ContinuousVariable("c2"),
             DiscreteVariable("d1", values=["a", "b"])],
            ContinuousVariable("y"),
            [ContinuousVariable("c3"),
             DiscreteVariable("d2", values=["c", "d"]),
             StringVariable("s1"),
             StringVariable("s2")]
        )
        metas = np.array(
            [0, 1, 0, 1, 1, np.nan, 1] +
            [0, 0, 0, 0, np.nan, 1, 1] +
            "a  b  c  d  e     f    g".split() +
            list("ABCDEF") + [""], dtype=object).reshape(-1, 7).T
        self.table = Table.from_numpy(
            self.domain,
            np.array(
                [[0, 0, 0],
                 [0, -1, 0],
                 [np.nan, 1, 0],
                 [1, 1, np.nan],
                 [1, 1, 1],
                 [1, 1, 1],
                 [1, 1, 1]]),
            np.array(
                [0, 1, 0, 1, np.nan, 1, 1]),
            metas
        )

    def test_row_filters_is_defined(self):
        filtered = IsDefined()(self.table)
        self.assertEqual(list(filtered.metas[:, -2].flatten()), list("ab"))

        val_filter = Values([
            FilterContinuous(None, FilterContinuous.IsDefined)])
        filtered = val_filter(self.table)
        self.assertEqual(list(filtered.metas[:, -2].flatten()), list("abdg"))

        val_filter = Values([FilterString(None, FilterString.IsDefined)])
        filtered = val_filter(self.table)
        self.assertEqual(list(filtered.metas[:, -2].flatten()), list("abcdef"))

    def test_row_filter_no_discrete(self):
        val_filter = Values([FilterDiscrete(None, "a")])
        self.assertRaises(ValueError, val_filter, self.table)

    def test_row_filter_continuous(self):
        val_filter = Values([
            FilterContinuous(None, FilterContinuous.GreaterEqual, 0)])
        filtered = val_filter(self.table)
        self.assertEqual(list(filtered.metas[:, -2].flatten()), list("adg"))

        val_filter = Values([
            FilterContinuous(None, FilterContinuous.Greater, 0)])
        filtered = val_filter(self.table)
        self.assertEqual(list(filtered.metas[:, -2].flatten()), list("dg"))

        val_filter = Values([
            FilterContinuous(None, FilterContinuous.Less, 1)])
        filtered = val_filter(self.table)
        self.assertEqual(list(filtered.metas[:, -2].flatten()), ["a"])

    def test_row_filter_string(self):
        self.table.metas[:, -1] = self.table.metas[::-1, -2]
        val_filter = Values([
            FilterString(None, FilterString.Between, "c", "e")])
        filtered = val_filter(self.table)
        self.assertEqual(list(filtered.metas[:, -2].flatten()), list("cde"))

    def test_row_stringlist(self):
        val_filter = Values([
            FilterStringList(None, list("bBdDe"))])
        filtered = val_filter(self.table)
        self.assertEqual(list(filtered.metas[:, -2].flatten()), list("bd"))

        val_filter = Values([
            FilterStringList(None, list("bDe"), case_sensitive=False)])
        filtered = val_filter(self.table)
        self.assertEqual(list(filtered.metas[:, -2].flatten()), list("bde"))

    def test_row_stringregex(self):
        val_filter = Values([FilterRegex(None, "[bBdDe]")])
        filtered = val_filter(self.table)
        self.assertEqual(list(filtered.metas[:, -2].flatten()), list("bd"))

    def test_is_defined(self):
        val_filter = IsDefined(columns=["c3"])
        filtered = val_filter(self.table)
        self.assertEqual(list(filtered.metas[:, -2].flatten()), list("abcdeg"))


if __name__ == "__main__":
    unittest.main()
