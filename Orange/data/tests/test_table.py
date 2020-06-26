import unittest

import numpy as np
import scipy.sparse as sp

from Orange.data import (
    ContinuousVariable, DiscreteVariable, StringVariable,
    Domain, Table, IsDefined, FilterContinuous, Values, FilterString,
    FilterDiscrete, FilterStringList, FilterRegex)
from Orange.util import OrangeDeprecationWarning


class TestTableInit(unittest.TestCase):
    def test_empty_table(self):
        t = Table()
        self.assertEqual(t.domain.attributes, ())
        self.assertEqual(t.X.shape, (0, 0))
        self.assertEqual(t.Y.shape, (0, 0))
        self.assertEqual(t.W.shape, (0, 0))
        self.assertEqual(t.metas.shape, (0, 0))
        self.assertEqual(t.ids.shape, (0, ))
        self.assertEqual(t.attributes, {})

    def test_warnings(self):
        domain = Domain([ContinuousVariable("x")])
        self.assertWarns(OrangeDeprecationWarning, Table, domain)
        self.assertWarns(OrangeDeprecationWarning, Table, domain, Table())
        self.assertWarns(OrangeDeprecationWarning, Table, domain, [[12]])
        self.assertWarns(OrangeDeprecationWarning, Table, np.zeros((5, 5)))

    def test_invalid_call_with_kwargs(self):
        self.assertRaises(TypeError, Table, Y=[])
        self.assertRaises(TypeError, Table, "iris", 42)
        self.assertRaises(TypeError, Table, Table(), 42)

    def test_from_numpy(self):
        X = np.arange(20).reshape(5, 4)
        Y = np.arange(5) % 2
        metas = np.array(list("abcde")).reshape(5, 1)
        W = np.arange(5) / 5
        ids = np.arange(100, 105, dtype=np.int)
        attributes = dict(a=5, b="foo")

        dom = Domain([ContinuousVariable(x) for x in "abcd"],
                     DiscreteVariable("e", values=("no", "yes")),
                     [StringVariable("s")])

        for func in (Table.from_numpy, Table):
            table = func(dom, X, Y, metas, W, attributes, ids)
            np.testing.assert_equal(X, table.X)
            np.testing.assert_equal(Y, table.Y)
            np.testing.assert_equal(metas, table.metas)
            np.testing.assert_equal(W, table.W)
            self.assertEqual(attributes, table.attributes)
            np.testing.assert_equal(ids, table.ids)

            table = func(dom, X, Y, metas, W)
            np.testing.assert_equal(X, table.X)
            np.testing.assert_equal(Y, table.Y)
            np.testing.assert_equal(metas, table.metas)
            np.testing.assert_equal(W, table.W)
            self.assertEqual(ids.shape, (5, ))

            table = func(dom, X, Y, metas)
            np.testing.assert_equal(X, table.X)
            np.testing.assert_equal(Y, table.Y)
            np.testing.assert_equal(metas, table.metas)
            self.assertEqual(table.W.shape, (5, 0))
            self.assertEqual(table.ids.shape, (5, ))

            table = func(Domain(dom.attributes, dom.class_var), X, Y)
            np.testing.assert_equal(X, table.X)
            np.testing.assert_equal(Y, table.Y)
            self.assertEqual(table.metas.shape, (5, 0))
            self.assertEqual(table.W.shape, (5, 0))
            self.assertEqual(table.ids.shape, (5, ))

            table = func(Domain(dom.attributes), X)
            np.testing.assert_equal(X, table.X)
            self.assertEqual(table.Y.shape, (5, 0))
            self.assertEqual(table.metas.shape, (5, 0))
            self.assertEqual(table.W.shape, (5, 0))
            self.assertEqual(table.ids.shape, (5, ))

            self.assertRaises(ValueError, func, dom, X, Y, metas, W[:4])
            self.assertRaises(ValueError, func, dom, X, Y, metas[:4])
            self.assertRaises(ValueError, func, dom, X, Y[:4])

    def test_from_numpy_sparse(self):
        domain = Domain([ContinuousVariable(c) for c in "abc"])
        x = np.arange(12).reshape(4, 3)

        t = Table.from_numpy(domain, x, None, None)
        self.assertFalse(sp.issparse(t.X))

        t = Table.from_numpy(domain, sp.csr_matrix(x))
        self.assertTrue(sp.isspmatrix_csr(t.X))

        t = Table.from_numpy(domain, sp.csc_matrix(x))
        self.assertTrue(sp.isspmatrix_csc(t.X))

        t = Table.from_numpy(domain, sp.coo_matrix(x))
        self.assertTrue(sp.isspmatrix_csr(t.X))

        t = Table.from_numpy(domain, sp.lil_matrix(x))
        self.assertTrue(sp.isspmatrix_csr(t.X))

        t = Table.from_numpy(domain, sp.bsr_matrix(x))
        self.assertTrue(sp.isspmatrix_csr(t.X))


class TestTableFilters(unittest.TestCase):
    def setUp(self):
        self.domain = Domain(
            [ContinuousVariable("c1"),
             ContinuousVariable("c2"),
             DiscreteVariable("d1", values=("a", "b"))],
            ContinuousVariable("y"),
            [ContinuousVariable("c3"),
             DiscreteVariable("d2", values=("c", "d")),
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

        val_filter = Values([IsDefined()])
        filtered = val_filter(self.table)
        self.assertEqual(list(filtered.metas[:, -2].flatten()), list("ab"))

        val_filter = Values([IsDefined(negate=True)])
        filtered = val_filter(self.table)
        self.assertEqual(list(filtered.metas[:, -2].flatten()), list("cdefg"))

        val_filter = Values([IsDefined(["c1"])])
        filtered = val_filter(self.table)
        self.assertEqual(list(filtered.metas[:, -2].flatten()), list("abdefg"))

        val_filter = Values([IsDefined(["c1"], negate=True)])
        filtered = val_filter(self.table)
        self.assertEqual(list(filtered.metas[:, -2].flatten()), list("c"))

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
