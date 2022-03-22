import pickle
import unittest
import os
from distutils.version import LooseVersion

import numpy as np
import scipy.sparse as sp

import Orange
from Orange.data import (
    ContinuousVariable, DiscreteVariable, StringVariable,
    Domain, Table, IsDefined, FilterContinuous, Values, FilterString,
    FilterDiscrete, FilterStringList, FilterRegex)
from Orange.util import OrangeDeprecationWarning


class TestTableInit(unittest.TestCase):

    def test_is_view_is_copy_deprecated(self):
        """This test is to be included in the 3.32 release and will fail in
        version 3.34. This serves as a reminder to remove the deprecated methods
        and this test."""
        if LooseVersion(Orange.__version__) >= LooseVersion("3.34"):
            self.fail(
                "`Orange.data.Table.is_view` and `Orange.data.Table.is_copy` "
                "were deprecated in version 3.32, and there have been two minor "
                "versions in between. Please remove the deprecated methods."
            )

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
        ids = np.arange(100, 105, dtype=int)
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

    @staticmethod
    def _new_table(attrs, classes, metas, s):
        def nz(x):  # pylint: disable=invalid-name
            return x if x.size else np.empty((5, 0))

        domain = Domain(attrs, classes, metas)
        X = np.arange(s, s + len(attrs) * 5).reshape(5, -1)
        Y = np.arange(100 + s, 100 + s + len(classes) * 5)
        if len(classes) > 1:
            Y = Y.reshape(5, -1)
        M = np.arange(200 + s, 200 + s + len(metas) * 5).reshape(5, -1)
        return Table.from_numpy(domain, nz(X), nz(Y), nz(M))

    def test_concatenate_horizontal(self):
        a, b, c, d, e, f, g = map(ContinuousVariable, "abcdefg")

        # Common case; one class, no empty's
        tab1 = self._new_table((a, b), (c, ), (d, ), 0)
        tab2 = self._new_table((e, ), (), (f, g), 1000)
        joined = Table.concatenate((tab1, tab2), axis=1)
        domain = joined.domain
        self.assertEqual(domain.attributes, (a, b, e))
        self.assertEqual(domain.class_vars, (c, ))
        self.assertEqual(domain.metas, (d, f, g))
        np.testing.assert_equal(joined.X, np.hstack((tab1.X, tab2.X)))
        np.testing.assert_equal(joined.Y, tab1.Y)
        np.testing.assert_equal(joined.metas, np.hstack((tab1.metas, tab2.metas)))

        # One part of one table is empty
        tab1 = self._new_table((a, b), (), (), 0)
        tab2 = self._new_table((), (), (c, ), 1000)
        joined = Table.concatenate((tab1, tab2), axis=1)
        domain = joined.domain
        self.assertEqual(domain.attributes, (a, b))
        self.assertEqual(domain.class_vars, ())
        self.assertEqual(domain.metas, (c, ))
        np.testing.assert_equal(joined.X, np.hstack((tab1.X, tab2.X)))
        np.testing.assert_equal(joined.metas, np.hstack((tab1.metas, tab2.metas)))

        # Multiple classes, two empty parts are merged
        tab1 = self._new_table((a, b), (c, ), (), 0)
        tab2 = self._new_table((), (d, ), (), 1000)
        joined = Table.concatenate((tab1, tab2), axis=1)
        domain = joined.domain
        self.assertEqual(domain.attributes, (a, b))
        self.assertEqual(domain.class_vars, (c, d))
        self.assertEqual(domain.metas, ())
        np.testing.assert_equal(joined.X, np.hstack((tab1.X, tab2.X)))
        np.testing.assert_equal(joined.Y, np.vstack((tab1.Y, tab2.Y)).T)

        # Merging of attributes and selection of weights
        tab1 = self._new_table((a, b), (c, ), (), 0)
        tab1.attributes = dict(a=5, b=7)
        tab2 = self._new_table((d, ), (e, ), (), 1000)
        with tab2.unlocked():
            tab2.W = np.arange(5)
        tab3 = self._new_table((f, g), (), (), 2000)
        tab3.attributes = dict(a=1, c=4)
        with tab3.unlocked():
            tab3.W = np.arange(5, 10)
        joined = Table.concatenate((tab1, tab2, tab3), axis=1)
        domain = joined.domain
        self.assertEqual(domain.attributes, (a, b, d, f, g))
        self.assertEqual(domain.class_vars, (c, e))
        self.assertEqual(domain.metas, ())
        np.testing.assert_equal(joined.X, np.hstack((tab1.X, tab2.X, tab3.X)))
        np.testing.assert_equal(joined.Y, np.vstack((tab1.Y, tab2.Y)).T)
        self.assertEqual(joined.attributes, dict(a=5, b=7, c=4))
        np.testing.assert_equal(joined.ids, tab1.ids)
        np.testing.assert_equal(joined.W, tab2.W)

        # Raise an exception when no tables are given
        self.assertRaises(ValueError, Table.concatenate, (), axis=1)

    def test_concatenate_invalid_axis(self):
        self.assertRaises(ValueError, Table.concatenate, (), axis=2)

    def test_concatenate_names(self):
        a, b, c, d, e, f, g = map(ContinuousVariable, "abcdefg")

        tab1 = self._new_table((a, ), (c, ), (d, ), 0)
        tab2 = self._new_table((e, ), (), (f, g), 1000)
        tab3 = self._new_table((b, ), (), (), 1000)
        tab2.name = "tab2"
        tab3.name = "tab3"

        joined = Table.concatenate((tab1, tab2, tab3), axis=1)
        self.assertEqual(joined.name, "tab2")

    def test_with_column(self):
        a, b, c, d, e, f, g = map(ContinuousVariable, "abcdefg")
        col = np.arange(9, 14)
        colr = col.reshape(5, -1)
        tab = self._new_table((a, b, c), (d, ), (e, f), 0)

        # Add to attributes
        tabw = tab.add_column(g, np.arange(9, 14))
        self.assertEqual(tabw.domain.attributes, (a, b, c, g))
        np.testing.assert_equal(tabw.X, np.hstack((tab.X, colr)))
        np.testing.assert_equal(tabw.Y, tab.Y)
        np.testing.assert_equal(tabw.metas, tab.metas)

        # Add to metas
        tabw = tab.add_column(g, np.arange(9, 14), to_metas=True)
        self.assertEqual(tabw.domain.metas, (e, f, g))
        np.testing.assert_equal(tabw.X, tab.X)
        np.testing.assert_equal(tabw.Y, tab.Y)
        np.testing.assert_equal(tabw.metas, np.hstack((tab.metas, colr)))

        # Add to empty attributes
        tab = self._new_table((), (d, ), (e, f), 0)
        tabw = tab.add_column(g, np.arange(9, 14))
        self.assertEqual(tabw.domain.attributes, (g, ))
        np.testing.assert_equal(tabw.X, colr)
        np.testing.assert_equal(tabw.Y, tab.Y)
        np.testing.assert_equal(tabw.metas, tab.metas)

        # Add to empty metas
        tab = self._new_table((a, b, c), (d, ), (), 0)
        tabw = tab.add_column(g, np.arange(9, 14), to_metas=True)
        self.assertEqual(tabw.domain.metas, (g, ))
        np.testing.assert_equal(tabw.X, tab.X)
        np.testing.assert_equal(tabw.Y, tab.Y)
        np.testing.assert_equal(tabw.metas, colr)

        # Pass values as a list
        tab = self._new_table((a, ), (d, ), (e, f), 0)
        tabw = tab.add_column(g, [4, 2, -1, 2, 5])
        self.assertEqual(tabw.domain.attributes, (a, g))
        np.testing.assert_equal(
            tabw.X, np.array([[0, 1, 2, 3, 4], [4, 2, -1, 2, 5]]).T)

        # Add non-primitives as metas; join `float` and `object` to `object`
        tab = self._new_table((a, ), (d, ), (e, f), 0)
        t = StringVariable("t")
        tabw = tab.add_column(t, list("abcde"))
        self.assertEqual(tabw.domain.attributes, (a, ))
        self.assertEqual(tabw.domain.metas, (e, f, t))
        np.testing.assert_equal(
            tabw.metas,
            np.hstack((tab.metas, np.array(list("abcde")).reshape(5, -1))))


class TestTableLocking(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.orig_locking = Table.LOCKING
        if os.getenv("CI"):
            assert Table.LOCKING
        else:
            Table.LOCKING = True

    @classmethod
    def tearDownClass(cls):
        Table.LOCKING = cls.orig_locking

    def setUp(self):
        a, b, c, d, e, f, g = map(ContinuousVariable, "abcdefg")
        domain = Domain([a, b, c], d, [e, f])
        self.table = Table.from_numpy(
            domain,
            np.random.random((5, 3)),
            np.random.random(5),
            np.random.random((5, 2)))

    def test_tables_are_locked(self):
        tab = self.table

        with self.assertRaises(ValueError):
            tab.X[0, 0] = 0
        with self.assertRaises(ValueError):
            tab.Y[0] = 0
        with self.assertRaises(ValueError):
            tab.metas[0, 0] = 0
        with self.assertRaises(ValueError):
            tab.W[0] = 0

        with self.assertRaises(ValueError):
            tab.X = np.random.random((5, 3))
        with self.assertRaises(ValueError):
            tab.Y = np.random.random(5)
        with self.assertRaises(ValueError):
            tab.metas = np.random.random((5, 2))
        with self.assertRaises(ValueError):
            tab.W = np.random.random(5)

    def test_unlocking(self):
        tab = self.table
        with tab.unlocked():
            tab.X[0, 0] = 0
            tab.Y[0] = 0
            tab.metas[0, 0] = 0

            tab.X = np.random.random((5, 3))
            tab.Y = np.random.random(5)
            tab.metas = np.random.random((5, 2))
            tab.W = np.random.random(5)

        with tab.unlocked(tab.Y):
            tab.Y[0] = 0
            with self.assertRaises(ValueError):
                tab.X[0, 0] = 0
            with tab.unlocked():
                tab.X[0, 0] = 0
            with self.assertRaises(ValueError):
                tab.X[0, 0] = 0

    def test_force_unlocking(self):
        tab = self.table
        with tab.unlocked():
            tab.Y = np.arange(10)[:5]

        # tab.Y is now a view and can't be unlocked
        with self.assertRaises(ValueError):
            with tab.unlocked(tab.X, tab.Y):
                pass
        # Tets that tab.X was not left unlocked
        with self.assertRaises(ValueError):
            tab.X[0, 0] = 0

        # This is not how force unlocking should be used! Force unlocking is
        # meant primarily for passing tables to Cython code that does not
        # properly define ndarrays as const. They should not modify the table;
        # modification here is meant only for testing.
        with tab.force_unlocked(tab.X, tab.Y):
            tab.X[0, 0] = 0
            tab.Y[0] = 0

    def test_locking_flag(self):
        try:
            default = Table.LOCKING
            Table.LOCKING = False
            self.setUp()
            self.table.X[0, 0] = 0
        finally:
            Table.LOCKING = default

    def test_unpickled_empty_weights(self):
        # ensure that unpickled empty arrays could be unlocked
        self.assertEqual(0, self.table.W.size)
        unpickled = pickle.loads(pickle.dumps(self.table))
        with unpickled.unlocked():
            pass

    def test_unpickling_resets_locks(self):
        default = Table.LOCKING
        try:
            self.setUp()
            pickled_locked = pickle.dumps(self.table)
            Table.LOCKING = False
            tab = pickle.loads(pickled_locked)
            tab.X[0, 0] = 1
            Table.LOCKING = True
            tab = pickle.loads(pickled_locked)
            with self.assertRaises(ValueError):
                tab.X[0, 0] = 1
        finally:
            Table.LOCKING = default

    def test_unpickled_owns_data(self):
        try:
            default = Table.LOCKING
            Table.LOCKING = False
            self.setUp()
            table = self.table
            table.X = table.X.view()
        finally:
            Table.LOCKING = default

        unpickled = pickle.loads(pickle.dumps(table))
        self.assertTrue(all(ar.base is None
                            for ar in (unpickled.X, unpickled.Y, unpickled.W, unpickled.metas)))
        with unpickled.unlocked():
            unpickled.X[0, 0] = 42

    @staticmethod
    def test_unlock_table_derived():
        # pylint: disable=abstract-method
        class ExtendedTable(Table):
            pass

        t = ExtendedTable.from_file("iris")
        with t.unlocked():
            pass


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
            list("ABCDEF") + [""], dtype=object).reshape(-1, 7).T.copy()
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
        with self.table.unlocked():
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
