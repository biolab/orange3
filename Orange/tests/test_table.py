# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring

import copy
import os
import pickle
import random
import unittest
import warnings
from unittest.mock import Mock, MagicMock, patch
from itertools import chain
from math import isnan
from threading import Thread
from time import sleep, time

import numpy as np
import scipy.sparse as sp

from Orange import data
from Orange.data import (filter, Unknown, Variable, Table, DiscreteVariable,
                         ContinuousVariable, Domain, StringVariable)
from Orange.data.util import SharedComputeValue
from Orange.tests import test_dirname
from Orange.data.table import _optimize_indices


class TableTestCase(unittest.TestCase):
    def setUp(self):
        data.table.dataset_dirs.append(test_dirname())

    def tearDown(self):
        data.table.dataset_dirs.remove(test_dirname())

    def test_indexing_class(self):
        d = data.Table("datasets/test1")
        self.assertEqual([e.get_class() for e in d], ["t", "t", "f"])
        cind = len(d.domain.variables) - 1
        self.assertEqual([e[cind] for e in d], ["t", "t", "f"])
        self.assertEqual([e["d"] for e in d], ["t", "t", "f"])
        cvar = d.domain.class_var
        self.assertEqual([e[cvar] for e in d], ["t", "t", "f"])

    def test_filename(self):
        dir = data.table.get_sample_datasets_dir()
        d = data.Table("iris")
        self.assertEqual(d.__file__, os.path.join(dir, "iris.tab"))

        d = data.Table("datasets/test2.tab")
        self.assertTrue(d.__file__.endswith("test2.tab"))  # platform dependent

    def test_indexing(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d = data.Table("datasets/test2")

            # regular, discrete
            varc = d.domain["c"]
            self.assertEqual(d[0, 1], "0")
            self.assertEqual(d[0, varc], "0")
            self.assertEqual(d[0, "c"], "0")
            self.assertEqual(d[0][1], "0")
            self.assertEqual(d[0][varc], "0")
            self.assertEqual(d[0]["c"], "0")
            self.assertEqual(d[np.int_(0), np.int_(1)], "0")
            self.assertEqual(d[np.int_(0)][np.int_(1)], "0")

            # regular, continuous
            varb = d.domain["b"]
            self.assertEqual(d[0, 0], 0)
            self.assertEqual(d[0, varb], 0)
            self.assertEqual(d[0, "b"], 0)
            self.assertEqual(d[0][0], 0)
            self.assertEqual(d[0][varb], 0)
            self.assertEqual(d[0]["b"], 0)
            self.assertEqual(d[np.int_(0), np.int_(0)], 0)
            self.assertEqual(d[np.int_(0)][np.int_(0)], 0)

            # negative
            varb = d.domain["b"]
            self.assertEqual(d[-2, 0], 3.333)
            self.assertEqual(d[-2, varb], 3.333)
            self.assertEqual(d[-2, "b"], 3.333)
            self.assertEqual(d[-2][0], 3.333)
            self.assertEqual(d[-2][varb], 3.333)
            self.assertEqual(d[-2]["b"], 3.333)
            self.assertEqual(d[np.int_(-2), np.int_(0)], 3.333)
            self.assertEqual(d[np.int_(-2)][np.int_(0)], 3.333)

            # meta, discrete
            vara = d.domain["a"]
            metaa = d.domain.index("a")
            self.assertEqual(d[0, metaa], "A")
            self.assertEqual(d[0, vara], "A")
            self.assertEqual(d[0, "a"], "A")
            self.assertEqual(d[0][metaa], "A")
            self.assertEqual(d[0][vara], "A")
            self.assertEqual(d[0]["a"], "A")
            self.assertEqual(d[np.int_(0), np.int_(metaa)], "A")
            self.assertEqual(d[np.int_(0)][np.int_(metaa)], "A")

            # meta, string
            vare = d.domain["e"]
            metae = d.domain.index("e")
            self.assertEqual(d[0, metae], "i")
            self.assertEqual(d[0, vare], "i")
            self.assertEqual(d[0, "e"], "i")
            self.assertEqual(d[0][metae], "i")
            self.assertEqual(d[0][vare], "i")
            self.assertEqual(d[0]["e"], "i")
            self.assertEqual(d[np.int_(0), np.int_(metae)], "i")
            self.assertEqual(d[np.int_(0)][np.int_(metae)], "i")

    def test_indexing_example(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d = data.Table("datasets/test2")
            e = d[0]

            # regular, discrete
            varc = d.domain["c"]
            self.assertEqual(e[1], "0")
            self.assertEqual(e[varc], "0")
            self.assertEqual(e["c"], "0")
            self.assertEqual(e[np.int_(1)], "0")

            # regular, continuous
            varb = d.domain["b"]
            self.assertEqual(e[0], 0)
            self.assertEqual(e[varb], 0)
            self.assertEqual(e["b"], 0)
            self.assertEqual(e[np.int_(0)], 0)

            # meta, discrete
            vara = d.domain["a"]
            metaa = d.domain.index("a")
            self.assertEqual(e[metaa], "A")
            self.assertEqual(e[vara], "A")
            self.assertEqual(e["a"], "A")
            self.assertEqual(e[np.int_(metaa)], "A")

            # meta, string
            vare = d.domain["e"]
            metae = d.domain.index("e")
            self.assertEqual(e[metae], "i")
            self.assertEqual(e[vare], "i")
            self.assertEqual(e["e"], "i")
            self.assertEqual(e[np.int_(metae)], "i")

    def test_indexing_assign_value(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d = data.Table("datasets/test2")

            # meta
            vara = d.domain["a"]
            metaa = d.domain.index("a")

            self.assertEqual(d[0, "a"], "A")

            with d.unlocked():
                d[0, "a"] = "B"
            self.assertEqual(d[0, "a"], "B")
            with d.unlocked():
                d[0]["a"] = "A"
            self.assertEqual(d[0, "a"], "A")

            with d.unlocked():
                d[0, vara] = "B"
            self.assertEqual(d[0, "a"], "B")
            with d.unlocked():
                d[0][vara] = "A"
            self.assertEqual(d[0, "a"], "A")

            with d.unlocked():
                d[0, metaa] = "B"
            self.assertEqual(d[0, "a"], "B")
            with d.unlocked():
                d[0][metaa] = "A"
            self.assertEqual(d[0, "a"], "A")

            with d.unlocked():
                d[0, np.int_(metaa)] = "B"
            self.assertEqual(d[0, "a"], "B")
            with d.unlocked():
                d[0][np.int_(metaa)] = "A"
            self.assertEqual(d[0, "a"], "A")

            # regular
            varb = d.domain["b"]

            self.assertEqual(d[0, "b"], 0)
            with d.unlocked():
                d[0, "b"] = 42
            self.assertEqual(d[0, "b"], 42)
            with d.unlocked():
                d[0]["b"] = 0
            self.assertEqual(d[0, "b"], 0)

            with d.unlocked():
                d[0, varb] = 42
            self.assertEqual(d[0, "b"], 42)
            with d.unlocked():
                d[0][varb] = 0
            self.assertEqual(d[0, "b"], 0)

            with d.unlocked():
                d[0, 0] = 42
            self.assertEqual(d[0, "b"], 42)
            with d.unlocked():
                d[0][0] = 0
            self.assertEqual(d[0, "b"], 0)

            with d.unlocked():
                d[0, np.int_(0)] = 42
            self.assertEqual(d[0, "b"], 42)
            with d.unlocked():
                d[0][np.int_(0)] = 0
            self.assertEqual(d[0, "b"], 0)

    def test_indexing_assign_example(self):
        def almost_equal_list(s, t):
            for e, f in zip(s, t):
                self.assertAlmostEqual(e, f)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d = data.Table("datasets/test2")

            self.assertFalse(isnan(d[0, "a"]))
            with d.unlocked():
                d[0] = ["3.14", "1", "f"]
            almost_equal_list(d[0].values(), [3.14, "1", "f"])
            self.assertTrue(isnan(d[0, "a"]))

            with d.unlocked():
                d[0] = [3.15, 1, "t"]
            almost_equal_list(d[0].values(), [3.15, "0", "t"])

            with d.unlocked():
                d[np.int_(0)] = [3.15, 2, "f"]
            almost_equal_list(d[0].values(), [3.15, 2, "f"])

            with d.unlocked(), self.assertRaises(ValueError):
                d[0] = ["3.14", "1"]

            with d.unlocked(), self.assertRaises(ValueError):
                d[np.int_(0)] = ["3.14", "1"]

            ex = data.Instance(d.domain, ["3.16", "1", "f"])
            with d.unlocked():
                d[0] = ex
            almost_equal_list(d[0].values(), [3.16, "1", "f"])

            ex = data.Instance(d.domain, ["3.16", 2, "t"])
            with d.unlocked():
                d[np.int_(0)] = ex
            almost_equal_list(d[0].values(), [3.16, 2, "t"])

            ex = data.Instance(d.domain, ["3.16", "1", "f"])
            ex["e"] = "mmmapp"
            with d.unlocked():
                d[0] = ex
            almost_equal_list(d[0].values(), [3.16, "1", "f"])
            self.assertEqual(d[0, "e"], "mmmapp")

    def test_slice(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d = data.Table("datasets/test2")
            x = d[:3]
            self.assertEqual(len(x), 3)
            self.assertEqual([e[0] for e in x], [0, 1.1, 2.22])

            x = d[2:5]
            self.assertEqual(len(x), 3)
            self.assertEqual([e[0] for e in x], [2.22, 2.23, 2.24])

            x = d[4:1:-1]
            self.assertEqual(len(x), 3)
            self.assertEqual([e[0] for e in x], [2.24, 2.23, 2.22])

            x = d[-3:]
            self.assertEqual(len(x), 3)
            self.assertEqual([e[0] for e in x], [2.26, 3.333, Unknown])

    def test_assign_slice_value(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d = data.Table("datasets/test2")
            with d.unlocked():
                d[2:5, 0] = 42
            self.assertEqual([e[0] for e in d],
                             [0, 1.1, 42, 42, 42, 2.25, 2.26, 3.333, Unknown])
            with d.unlocked():
                d[:3, "b"] = 43
            self.assertEqual([e[0] for e in d],
                             [43, 43, 43, 42, 42, 2.25, 2.26, 3.333, None])
            with d.unlocked():
                d[-2:, d.domain[0]] = 44
            self.assertEqual([e[0] for e in d],
                             [43, 43, 43, 42, 42, 2.25, 2.26, 44, 44])

            with d.unlocked():
                d[2:5, "a"] = "A"
            self.assertEqual([e["a"] for e in d], list("ABAAACCDE"))

    def test_multiple_indices(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d = data.Table("datasets/test2")

            with self.assertRaises(IndexError):
                x = d[2, 5, 1]

            with self.assertRaises(IndexError):
                x = d[(2, 5, 1)]

            x = d[[2, 5, 1]]
            self.assertEqual([e[0] for e in x], [2.22, 2.25, 1.1])

    def test_assign_multiple_indices_value(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d = data.Table("datasets/test2")

            with d.unlocked():
                d[1:4, "b"] = 42
            self.assertEqual([e[0] for e in d],
                             [0, 42, 42, 42, 2.24, 2.25, 2.26, 3.333, None])

            with d.unlocked():
                d[range(5, 2, -1), "b"] = None
            self.assertEqual([e[d.domain[0]] for e in d],
                             [0, 42, 42, None, "?", "", 2.26, 3.333, None])

    def test_set_multiple_indices_example(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d = data.Table("datasets/test2")

            vals = [e[0] for e in d]
            with d.unlocked():
                d[[1, 2, 5]] = [42, None, None]
            vals[1] = vals[2] = vals[5] = 42
            self.assertEqual([e[0] for e in d], vals)

    def test_bool(self):
        d = data.Table("iris")
        self.assertTrue(d)

        d = data.Table("datasets/test3")
        self.assertFalse(d)

        d = data.Table("iris")
        self.assertTrue(d)

    def test_checksum(self):
        d = data.Table("zoo")
        with d.unlocked():
            d[42, 3] = 0
        crc1 = d.checksum(False)

        with d.unlocked():
            d[42, 3] = 1
        crc2 = d.checksum(False)
        self.assertNotEqual(crc1, crc2)

        with d.unlocked():
            d[42, 3] = 0
        crc3 = d.checksum(False)
        self.assertEqual(crc1, crc3)

        _ = d[42, "name"]
        with d.unlocked():
            d[42, "name"] = "non-animal"
        crc4 = d.checksum(False)
        self.assertEqual(crc1, crc4)
        crc4 = d.checksum(True)
        crc5 = d.checksum(1)
        crc6 = d.checksum(False)
        self.assertNotEqual(crc1, crc4)
        self.assertNotEqual(crc1, crc5)
        self.assertEqual(crc1, crc6)

    def test_total_weight(self):
        d = data.Table("zoo")
        self.assertEqual(d.total_weight(), len(d))

        with d.unlocked():
            d.set_weights(0)
            d[0].weight = 0.1
            d[10].weight = 0.2
            d[-1].weight = 0.3
        self.assertAlmostEqual(d.total_weight(), 0.6)

    def test_has_missing(self):
        d = data.Table("zoo")
        self.assertFalse(d.has_missing())
        self.assertFalse(d.has_missing_class())

        with d.unlocked():
            d[10, 3] = "?"
        self.assertTrue(d.has_missing())
        self.assertFalse(d.has_missing_class())

        with d.unlocked():
            d[10].set_class("?")
        self.assertTrue(d.has_missing())
        self.assertTrue(d.has_missing_class())

        with d.unlocked():
            d = data.Table("datasets/test3")
        self.assertFalse(d.has_missing())
        self.assertFalse(d.has_missing_class())

    def test_shuffle(self):
        d = data.Table("zoo")
        crc = d.checksum()
        names = set(str(x["name"]) for x in d)
        ids = d.ids

        with d.unlocked_reference():
            d.shuffle()
        self.assertNotEqual(crc, d.checksum())
        self.assertSetEqual(names, set(str(x["name"]) for x in d))
        self.assertTrue(np.any(ids - d.ids != 0))
        crc2 = d.checksum()

        x = d[2:10]
        crcx = x.checksum()
        with d.unlocked_reference():
            d.shuffle()
        self.assertNotEqual(crc2, d.checksum())
        self.assertEqual(crcx, x.checksum())

        crc2 = d.checksum()
        with x.unlocked_reference():
            x.shuffle()
        self.assertNotEqual(crcx, x.checksum())
        self.assertEqual(crc2, d.checksum())
        self.assertLess(set(x.ids), set(ids))

    @staticmethod
    def not_less_ex(ex1, ex2):
        for v1, v2 in zip(ex1, ex2):
            if v1 != v2:
                return v1 < v2
        return True

    @staticmethod
    def sorted(d):
        for i in range(1, len(d)):
            if not TableTestCase.not_less_ex(d[i - 1], d[i]):
                return False
        return True

    @staticmethod
    def not_less_ex_ord(ex1, ex2, ord):
        for a in ord:
            if ex1[a] != ex2[a]:
                return ex1[a] < ex2[a]
        return True

    @staticmethod
    def sorted_ord(d, ord):
        for i in range(1, len(d)):
            if not TableTestCase.not_less_ex_ord(d[i - 1], d[i], ord):
                return False
        return True

    def test_copy(self):
        t = data.Table.from_numpy(
            None, np.zeros((5, 3)), np.arange(5), np.zeros((5, 3)))

        copy = t.copy()
        self.assertTrue(np.all(t.X == copy.X))
        self.assertTrue(np.all(t.Y == copy.Y))
        self.assertTrue(np.all(t.metas == copy.metas))
        with copy.unlocked():
            copy[0] = [1, 1, 1, 1]
        self.assertFalse(np.all(t.X == copy.X))
        self.assertFalse(np.all(t.Y == copy.Y))
        self.assertFalse(np.all(t.metas == copy.metas))

    def test_copy_sparse(self):
        t = data.Table('iris').to_sparse()
        copy = t.copy()

        self.assertEqual((t.X != copy.X).nnz, 0)      # sparse matrices match by content
        np.testing.assert_equal(t.Y, copy.Y)
        np.testing.assert_equal(t.metas, copy.metas)

        self.assertNotEqual(id(t.X), id(copy.X))
        self.assertNotEqual(id(t._Y), id(copy._Y))
        self.assertNotEqual(id(t.metas), id(copy.metas))

        # ensure that copied sparse arrays do not share data
        with t.unlocked():
            t.X[0, 0] = 42
        self.assertEqual(copy.X[0, 0], 5.1)

    def test_concatenate(self):
        d1 = data.Domain(
            [data.ContinuousVariable(n) for n in "abc"],
            data.DiscreteVariable("y", values="ABC"),
            [data.StringVariable(n) for n in ["m1", "m2"]],
        )
        x1 = np.arange(6).reshape(2, 3)
        y1 = np.array([0, 1])
        m1 = np.array([["foo", "bar"], ["baz", "qux"]])
        w1 = np.random.random((2,))
        t1 = data.Table.from_numpy(d1, x1, y1, m1, w1)
        t1.ids = ids1 = np.array([100, 101])
        t1.attributes = {"a": 42, "c": 43}

        x2 = np.arange(6, 15).reshape(3, 3)
        y2 = np.array([1, 2, 0])
        m2 = np.array([["a", "b"], ["c", "d"], ["e", "f"]])
        w2 = np.random.random((3,))
        t2 = data.Table.from_numpy(d1, x2, y2, m2, w2)
        t2.ids = ids2 = np.array([102, 103, 104])
        t2.name = "t2"
        t2.attributes = {"a": 44, "b": 45}

        x3 = np.arange(15, 27).reshape(4, 3)
        y3 = np.array([2, 1, 1, 0])
        m3 = np.array([["g", "h"], ["i", "j"], ["k", "l"], ["m", "n"]])
        w3 = np.random.random((4,))
        t3 = data.Table.from_numpy(d1, x3, y3, m3, w3)
        t3.ids = ids3 = np.array([102, 103, 104, 105])
        t3.name = "t3"

        t1b = data.Table.concatenate((t1,))
        self.assertEqual(t1b.domain, t1.domain)
        np.testing.assert_almost_equal(t1b.X, x1)
        np.testing.assert_almost_equal(t1b.Y, y1)
        self.assertEqual(list(t1b.metas.flatten()),
                         list(m1.flatten()))
        np.testing.assert_almost_equal(t1b.W, w1)
        np.testing.assert_almost_equal(t1b.ids, ids1)
        self.assertEqual(t1b.name, t1.name)
        self.assertEqual(t1b.attributes, {"a": 42, "c": 43})

        t12 = data.Table.concatenate((t1, t2))
        self.assertEqual(t12.domain, t1.domain)
        np.testing.assert_almost_equal(t12.X, np.vstack((x1, x2)))
        np.testing.assert_almost_equal(t12.Y, np.hstack((y1, y2)))
        self.assertEqual(list(t12.metas.flatten()),
                         list(np.vstack((m1, m2)).flatten()))
        np.testing.assert_almost_equal(t12.W, np.hstack((w1, w2)))
        np.testing.assert_almost_equal(t12.ids, np.hstack((ids1, ids2)))
        self.assertEqual(t12.name, "t2")
        self.assertEqual(t12.attributes, {"a": 42, "c": 43, "b": 45})

        t123 = data.Table.concatenate((t1, t2, t3))
        self.assertEqual(t123.domain, t1.domain)
        np.testing.assert_almost_equal(t123.X, np.vstack((x1, x2, x3)))
        np.testing.assert_almost_equal(t123.Y, np.hstack((y1, y2, y3)))
        self.assertEqual(list(t123.metas.flatten()),
                         list(np.vstack((m1, m2, m3)).flatten()))
        np.testing.assert_almost_equal(t123.W, np.hstack((w1, w2, w3)))
        np.testing.assert_almost_equal(t123.ids, np.hstack((ids1, ids2, ids3)))
        self.assertEqual(t123.name, "t2")
        self.assertEqual(t123.attributes, {"a": 42, "c": 43, "b": 45})

        with t2.unlocked(t2.Y):
            t2.Y = np.atleast_2d(t2.Y).T
        t12 = data.Table.concatenate((t1, t2))
        self.assertEqual(t12.domain, t1.domain)
        np.testing.assert_almost_equal(t12.X, np.vstack((x1, x2)))
        np.testing.assert_almost_equal(t12.Y, np.hstack((y1, y2)))
        self.assertEqual(list(t12.metas.flatten()),
                         list(np.vstack((m1, m2)).flatten()))
        np.testing.assert_almost_equal(t12.W, np.hstack((w1, w2)))
        np.testing.assert_almost_equal(t12.ids, np.hstack((ids1, ids2)))
        self.assertEqual(t12.name, "t2")
        self.assertEqual(t12.attributes, {"a": 42, "c": 43, "b": 45})

    def test_concatenate_exceptions(self):
        zoo = data.Table("zoo")
        iris = data.Table("iris")

        self.assertRaises(ValueError, data.Table.concatenate, [])
        self.assertRaises(ValueError, data.Table.concatenate, [zoo, iris])

    def test_concatenate_sparse(self):
        iris = Table("iris")
        with iris.unlocked():
            iris.X = sp.csc_matrix(iris.X)
        new = Table.concatenate([iris, iris])
        self.assertEqual(len(new), 300)
        self.assertTrue(sp.issparse(new.X), "Concatenated X is not sparse.")
        self.assertFalse(sp.issparse(new.Y), "Concatenated Y is not dense.")
        self.assertFalse(sp.issparse(new.metas), "Concatenated metas is not dense.")
        self.assertEqual(len(new.ids), 300)

    def test_pickle(self):
        d = data.Table("zoo")
        s = pickle.dumps(d)
        d2 = pickle.loads(s)
        self.assertEqual(d[0], d2[0])

        self.assertEqual(d.checksum(include_metas=False),
                         d2.checksum(include_metas=False))

        d = data.Table("iris")
        s = pickle.dumps(d)
        d2 = pickle.loads(s)
        self.assertEqual(d[0], d2[0])
        self.assertEqual(d.checksum(include_metas=False),
                         d2.checksum(include_metas=False))

    def test_pickle_setstate(self):
        d = data.Table("zoo")
        s = pickle.dumps(d)
        with patch("Orange.data.Table.__setstate__", Mock()) as mock:
            pickle.loads(s)
            state = mock.call_args[0][0]
            for k in ["X", "_Y", "metas"]:
                self.assertIn(k, state)
                self.assertEqual(state[k].ndim, 2)
            self.assertIn("W", state)
            for k in ["_X", "Y", "_metas", "_W"]:
                self.assertNotIn(k, state)


    def test_translate_through_slice(self):
        d = data.Table("iris")
        dom = data.Domain(["petal length", "sepal length", "iris"],
                          source=d.domain)
        d_ref = d[:10, dom.variables]
        self.assertEqual(d_ref.domain.class_var, d.domain.class_var)
        self.assertEqual(d_ref[0, "petal length"], d[0, "petal length"])
        self.assertEqual(d_ref[0, "sepal length"], d[0, "sepal length"])
        self.assertEqual(d_ref.X.shape, (10, 2))
        self.assertEqual(d_ref.Y.shape, (10,))

    def test_saveTab(self):
        d = data.Table("iris")[:3]
        d.save("test-save.tab")
        try:
            d2 = data.Table("test-save.tab")
            for e1, e2 in zip(d, d2):
                self.assertEqual(e1, e2)
        finally:
            os.remove("test-save.tab")
            os.remove("test-save.tab.metadata")

        dom = data.Domain([data.ContinuousVariable("a")])
        d = data.Table.from_list(dom, [[i] for i in range(3)])
        d.save("test-save.tab")
        try:
            d2 = data.Table("test-save.tab")
            self.assertEqual(len(d.domain.attributes), 1)
            self.assertEqual(d.domain.class_var, None)
            for i in range(3):
                self.assertEqual(d2[i], [i])
        finally:
            os.remove("test-save.tab")

        dom = data.Domain([data.ContinuousVariable("a")], None)
        d = data.Table.from_list(dom, [[i] for i in range(3)])
        d.save("test-save.tab")
        try:
            d2 = data.Table("test-save.tab")
            self.assertEqual(len(d.domain.attributes), 1)
            for i in range(3):
                self.assertEqual(d2[i], [i])
        finally:
            os.remove("test-save.tab")

        d = data.Table("zoo")
        d.save("test-zoo.tab")
        dd = data.Table("test-zoo")
        try:
            self.assertTupleEqual(d.domain.metas, dd.domain.metas,
                                  msg="Meta attributes don't match.")
            self.assertTupleEqual(d.domain.variables, dd.domain.variables,
                                  msg="Attributes don't match.")

            np.testing.assert_almost_equal(d.W, dd.W,
                                           err_msg="Weights don't match.")
            for i in range(10):
                for j in d.domain.variables:
                    self.assertEqual(d[i][j], dd[i][j])
        finally:
            os.remove("test-zoo.tab")
            os.remove("test-zoo.tab.metadata")

        d = data.Table("zoo")
        with d.unlocked():
            d.set_weights(range(len(d)))
        d.save("test-zoo-weights.tab")
        dd = data.Table("test-zoo-weights")
        try:
            self.assertTupleEqual(d.domain.metas, dd.domain.metas,
                                  msg="Meta attributes don't match.")
            self.assertTupleEqual(d.domain.variables, dd.domain.variables,
                                  msg="Attributes don't match.")

            np.testing.assert_almost_equal(d.W, dd.W,
                                           err_msg="Weights don't match.")
            for i in range(10):
                for j in d.domain.variables:
                    self.assertEqual(d[i][j], dd[i][j])
        finally:
            os.remove("test-zoo-weights.tab")
            os.remove("test-zoo-weights.tab.metadata")

    def test_save_pickle(self):
        table = data.Table("iris")
        try:
            table.save("iris.pickle")
            table2 = data.Table.from_file("iris.pickle")
            np.testing.assert_almost_equal(table.X, table2.X)
            np.testing.assert_almost_equal(table.Y, table2.Y)
            self.assertEqual(table.domain[0], table2.domain[0])
        finally:
            os.remove("iris.pickle")

    def test_from_numpy(self):
        a = np.arange(20, dtype="d").reshape((4, 5)).copy()
        m = np.arange(4, dtype="d").reshape((4, 1)).copy()
        a[:, -1] = [0, 0, 0, 1]
        dom = data.Domain([data.ContinuousVariable(x) for x in "abcd"],
                          data.DiscreteVariable("e", values=("no", "yes")),
                          metas=[data.ContinuousVariable(x) for x in "f"])
        table = data.Table(dom, a, metas=m)
        with table.unlocked():
            for i in range(4):
                self.assertEqual(table[i].get_class(), "no" if i < 3 else "yes")
                for j in range(5):
                    self.assertEqual(a[i, j], table[i, j])

        with table.unlocked(), self.assertRaises(IndexError):
            table[0, -6] = 5

    def test_filter_is_defined(self):
        d = data.Table("iris")
        with d.unlocked():
            d[1, 4] = Unknown
        self.assertTrue(isnan(d[1, 4]))
        with d.unlocked():
            d[140, 0] = Unknown
        e = filter.IsDefined()(d)
        self.assertEqual(len(e), len(d) - 2)
        self.assertEqual(e[0], d[0])
        self.assertEqual(e[1], d[2])
        self.assertEqual(e[147], d[149])
        self.assertTrue(d.has_missing())
        self.assertFalse(e.has_missing())

    def test_filter_has_class(self):
        d = data.Table("iris")
        with d.unlocked():
            d[1, 4] = Unknown
        self.assertTrue(isnan(d[1, 4]))
        with d.unlocked():
            d[140, 0] = Unknown
        e = filter.HasClass()(d)
        self.assertEqual(len(e), len(d) - 1)
        self.assertEqual(e[0], d[0])
        self.assertEqual(e[1], d[2])
        self.assertEqual(e[148], d[149])
        self.assertTrue(d.has_missing())
        self.assertTrue(e.has_missing())
        self.assertFalse(e.has_missing_class())

    def test_filter_random(self):
        d = data.Table("iris")
        e = filter.Random(50)(d)
        self.assertEqual(len(e), 50)
        e = filter.Random(50, negate=True)(d)
        self.assertEqual(len(e), 100)
        for _ in range(5):
            e = filter.Random(0.2)(d)
            self.assertEqual(len(e), 30)
            bc = np.bincount(np.array(e.Y[:], dtype=int))
            if min(bc) > 7:
                break
        else:
            self.fail("Filter returns too uneven distributions")

    def test_filter_values_nested(self):
        d = data.Table("iris")
        f1 = filter.FilterContinuous(d.columns.sepal_length,
                                     filter.FilterContinuous.Between,
                                     min=4.5, max=5.0)
        f2 = filter.FilterContinuous(d.columns.sepal_width,
                                     filter.FilterContinuous.Between,
                                     min=3.1, max=3.4)
        f3 = filter.FilterDiscrete(d.columns.iris, [0, 1])
        f = filter.Values([filter.Values([f1, f2], conjunction=False), f3])
        self.assertEqual(41, len(f(d)))

    def test_filter_string_works_for_numeric_columns(self):
        var = StringVariable("s")
        data = Table.from_list(Domain([], metas=[var]),
                               [[x] for x in range(21)])
        # 1, 2, 3, ..., 18, 19, 20

        fs = filter.FilterString
        filters = [
            ((fs.Greater, "5"), dict(rows=4)),
            # 6, 7, 8, 9
            ((fs.Between, "15", "2"), dict(rows=6)),
            # 15, 16, 17, 18, 19, 2
            ((fs.Contains, "2"), dict(rows=3)),
            # 2, 12, 20
        ]

        for args, expected in filters:
            f = fs(var, *args)
            filtered_data = filter.Values([f])(data)
            self.assertEqual(len(filtered_data), expected["rows"],
                             "{} returned wrong number of rows".format(args))

    def test_filter_value_continuous(self):
        d = data.Table("iris")
        col = d.X[:, 2]

        v = d.columns
        f = filter.FilterContinuous(v.petal_length,
                                    filter.FilterContinuous.Between,
                                    min=4.5, max=5.1)

        x = filter.Values([f])(d)
        self.assertTrue(np.all(x.X[:, 2] >= 4.5))
        self.assertTrue(np.all(x.X[:, 2] <= 5.1))
        self.assertEqual(sum((col >= 4.5) * (col <= 5.1)), len(x))

        f.ref = 5.1
        f.oper = filter.FilterContinuous.Equal
        x = filter.Values([f])(d)
        self.assertTrue(np.all(x.X[:, 2] == 5.1))
        self.assertEqual(sum(col == 5.1), len(x))

        f.oper = filter.FilterContinuous.NotEqual
        x = filter.Values([f])(d)
        self.assertTrue(np.all(x.X[:, 2] != 5.1))
        self.assertEqual(sum(col != 5.1), len(x))

        f.oper = filter.FilterContinuous.Less
        x = filter.Values([f])(d)
        self.assertTrue(np.all(x.X[:, 2] < 5.1))
        self.assertEqual(sum(col < 5.1), len(x))

        f.oper = filter.FilterContinuous.LessEqual
        x = filter.Values([f])(d)
        self.assertTrue(np.all(x.X[:, 2] <= 5.1))
        self.assertEqual(sum(col <= 5.1), len(x))

        f.oper = filter.FilterContinuous.Greater
        x = filter.Values([f])(d)
        self.assertTrue(np.all(x.X[:, 2] > 5.1))
        self.assertEqual(sum(col > 5.1), len(x))

        f.oper = filter.FilterContinuous.GreaterEqual
        x = filter.Values([f])(d)
        self.assertTrue(np.all(x.X[:, 2] >= 5.1))
        self.assertEqual(sum(col >= 5.1), len(x))

        f.oper = filter.FilterContinuous.Outside
        f.ref, f.max = 4.5, 5.1
        x = filter.Values([f])(d)
        for e in x:
            self.assertTrue(e[2] < 4.5 or e[2] > 5.1)
        self.assertEqual(sum((col < 4.5) + (col > 5.1)), len(x))

        f.oper = filter.FilterContinuous.IsDefined
        f.ref = f.max = None
        x = filter.Values([f])(d)
        self.assertEqual(len(x), len(d))

        with d.unlocked():
            d[:30, v.petal_length] = Unknown
        x = filter.Values([f])(d)
        self.assertEqual(len(x), len(d) - 30)

    def test_filter_value_continuous_args(self):
        d = data.Table("iris")
        col = d.X[:, 2]
        v = d.columns

        f = filter.FilterContinuous(v.petal_length,
                                    filter.FilterContinuous.Equal, ref=5.1)
        x = filter.Values([f])(d)
        self.assertTrue(np.all(x.X[:, 2] == 5.1))
        self.assertEqual(sum(col == 5.1), len(x))

        f = filter.FilterContinuous(2,
                                    filter.FilterContinuous.Equal, ref=5.1)
        self.assertTrue(np.all(x.X[:, 2] == 5.1))
        self.assertEqual(sum(col == 5.1), len(x))

        f = filter.FilterContinuous("petal length",
                                    filter.FilterContinuous.Equal, ref=5.1)
        self.assertTrue(np.all(x.X[:, 2] == 5.1))
        self.assertEqual(sum(col == 5.1), len(x))

        f = filter.FilterContinuous("sepal length",
                                    filter.FilterContinuous.Equal, ref=5.1)
        f.column = 2
        self.assertTrue(np.all(x.X[:, 2] == 5.1))
        self.assertEqual(sum(col == 5.1), len(x))

        f = filter.FilterContinuous("sepal length",
                                    filter.FilterContinuous.Equal, ref=5.1)
        f.column = v.petal_length
        self.assertTrue(np.all(x.X[:, 2] == 5.1))
        self.assertEqual(sum(col == 5.1), len(x))

        f = filter.FilterContinuous(v.petal_length,
                                    filter.FilterContinuous.Equal, ref=18)
        f.ref = 5.1
        x = filter.Values([f])(d)
        self.assertTrue(np.all(x.X[:, 2] == 5.1))
        self.assertEqual(sum(col == 5.1), len(x))

        f = filter.FilterContinuous(v.petal_length,
                                    filter.FilterContinuous.Equal, ref=18)
        f.ref = 5.1
        x = filter.Values([f])(d)
        self.assertTrue(np.all(x.X[:, 2] == 5.1))
        self.assertEqual(sum(col == 5.1), len(x))

    def test_valueFilter_discrete(self):
        d = data.Table("zoo")

        f = filter.FilterDiscrete(d.domain.class_var, values=[2, 3, 4])
        for e in filter.Values([f])(d):
            self.assertTrue(e.get_class() in [2, 3, 4])

        f.values = ["mammal"]
        for e in filter.Values([f])(d):
            self.assertEqual(e.get_class(), "mammal")

        f = filter.FilterDiscrete(d.domain.class_var, values=[2, "mammal"])
        for e in filter.Values([f])(d):
            self.assertTrue(e.get_class() in [2, "mammal"])

        f = filter.FilterDiscrete(d.domain.class_var, values=[2, "martian"])
        self.assertRaises(ValueError, d._filter_values, f)

        f = filter.FilterDiscrete(d.domain.class_var, values=(2, data.Table))
        self.assertRaises(TypeError, d._filter_values, f)

        v = d.columns
        f = filter.FilterDiscrete(v.hair, values=None)
        self.assertEqual(len(filter.Values([f])(d)), len(d))

        with d.unlocked():
            d[:5, v.hair] = Unknown
        self.assertEqual(len(filter.Values([f])(d)), len(d) - 5)

    def test_valueFilter_string_is_defined(self):
        d = data.Table("datasets/test9.tab")
        f = filter.FilterString(-5, filter.FilterString.IsDefined)
        x = filter.Values([f])(d)
        self.assertEqual(len(x), 7)

    def test_valueFilter_discrete_meta_is_defined(self):
        d = data.Table("datasets/test9.tab")
        f = filter.FilterDiscrete(-4, None)
        x = filter.Values([f])(d)
        self.assertEqual(len(x), 8)

    def test_valueFilter_string_case_sens(self):
        d = data.Table("zoo")
        col = d[:, "name"].metas[:, 0]

        f = filter.FilterString("name",
                                filter.FilterString.Equal, "girl")
        x = filter.Values([f])(d)
        self.assertEqual(len(x), 1)
        self.assertEqual(x[0, "name"], "girl")
        self.assertTrue(np.all(x.metas == "girl"))

        f.oper = f.NotEqual
        x = filter.Values([f])(d)
        self.assertEqual(len(x), len(d) - 1)
        self.assertTrue(np.all(x[:, "name"] != "girl"))

        f.oper = f.Less
        x = filter.Values([f])(d)
        self.assertEqual(len(x), sum(col < "girl"))
        self.assertTrue(np.all(x.metas < "girl"))

        f.oper = f.LessEqual
        x = filter.Values([f])(d)
        self.assertEqual(len(x), sum(col <= "girl"))
        self.assertTrue(np.all(x.metas <= "girl"))

        f.oper = f.Greater
        x = filter.Values([f])(d)
        self.assertEqual(len(x), sum(col > "girl"))
        self.assertTrue(np.all(x.metas > "girl"))

        f.oper = f.GreaterEqual
        x = filter.Values([f])(d)
        self.assertEqual(len(x), sum(col >= "girl"))
        self.assertTrue(np.all(x.metas >= "girl"))

        f.oper = f.Between
        f.max = "lion"
        x = filter.Values([f])(d)
        self.assertEqual(len(x), sum((col >= "girl") * (col <= "lion")))
        self.assertTrue(np.all(x.metas >= "girl"))
        self.assertTrue(np.all(x.metas <= "lion"))

        f.oper = f.Outside
        x = filter.Values([f])(d)
        self.assertEqual(len(x), sum(col < "girl") + sum(col > "lion"))
        self.assertTrue(np.all((x.metas < "girl") + (x.metas > "lion")))

        f.oper = f.Contains
        f.ref = "ea"
        x = filter.Values([f])(d)
        for e in x:
            self.assertTrue("ea" in e["name"])
        self.assertEqual(len(x), len([e for e in col if "ea" in e]))

        f.oper = f.StartsWith
        f.ref = "sea"
        x = filter.Values([f])(d)
        for e in x:
            self.assertTrue(str(e["name"]).startswith("sea"))
        self.assertEqual(len(x), len([e for e in col if e.startswith("sea")]))

        f.oper = f.EndsWith
        f.ref = "ion"
        x = filter.Values([f])(d)
        for e in x:
            self.assertTrue(str(e["name"]).endswith("ion"))
        self.assertEqual(len(x), len([e for e in col if e.endswith("ion")]))

    def test_valueFilter_string_case_insens(self):
        d = data.Table("zoo")
        with d.unlocked():
            d[d[:, "name"].metas[:, 0] == "girl", "name"] = "GIrl"

        col = d[:, "name"].metas[:, 0]

        f = filter.FilterString("name",
                                filter.FilterString.Equal, "giRL")
        f.case_sensitive = False
        x = filter.Values([f])(d)
        self.assertEqual(len(x), 1)
        self.assertEqual(x[0, "name"], "GIrl")
        self.assertTrue(np.all(x.metas == "GIrl"))

        f.oper = f.NotEqual
        x = filter.Values([f])(d)
        self.assertEqual(len(x), len(d) - 1)
        self.assertTrue(np.all(x[:, "name"] != "GIrl"))

        f.oper = f.Less
        f.ref = "CHiCKEN"
        x = filter.Values([f])(d)
        self.assertEqual(len(x), sum(col < "chicken") - 1)  # girl!
        self.assertTrue(np.all(x.metas < "chicken"))

        f.oper = f.LessEqual
        x = filter.Values([f])(d)
        self.assertEqual(len(x), sum(col <= "chicken") - 1)
        self.assertTrue(np.all(x.metas <= "chicken"))

        f.oper = f.Greater
        x = filter.Values([f])(d)
        self.assertEqual(len(x), sum(col > "chicken") + 1)
        for e in x:
            self.assertGreater(str(e["name"]).lower(), "chicken")

        f.oper = f.GreaterEqual
        x = filter.Values([f])(d)
        self.assertEqual(len(x), sum(col >= "chicken") + 1)
        for e in x:
            self.assertGreaterEqual(str(e["name"]).lower(), "chicken")

        f.oper = f.Between
        f.max = "liOn"
        x = filter.Values([f])(d)
        self.assertEqual(len(x), sum((col >= "chicken") * (col <= "lion")) + 1)
        for e in x:
            self.assertTrue("chicken" <= str(e["name"]).lower() <= "lion")

        f.oper = f.Outside
        x = filter.Values([f])(d)
        self.assertEqual(len(x), sum(col < "chicken") + sum(col > "lion") - 1)
        self.assertTrue(np.all((x.metas < "chicken") + (x.metas > "lion")))

        f.oper = f.Contains
        f.ref = "iR"
        x = filter.Values([f])(d)
        for e in x:
            self.assertTrue("ir" in str(e["name"]).lower())
        self.assertEqual(len(x), len([e for e in col if "ir" in e]) + 1)

        f.oper = f.StartsWith
        f.ref = "GI"
        x = filter.Values([f])(d)
        for e in x:
            self.assertTrue(str(e["name"]).lower().startswith("gi"))
        self.assertEqual(len(x),
                         len([e for e in col if e.lower().startswith("gi")]))

        f.oper = f.EndsWith
        f.ref = "ion"
        x = filter.Values([f])(d)
        for e in x:
            self.assertTrue(str(e["name"]).endswith("ion"))
            self.assertTrue(str(e["name"]).endswith("ion"))
        self.assertEqual(len(x), len([e for e in col if e.endswith("ion")]))

    def test_valueFilter_regex(self):
        d = data.Table("zoo")
        f = filter.FilterRegex(d.domain['name'], '^c...$')
        x = filter.Values([f])(d)
        self.assertEqual(len(x), 7)

    def test_valueFilter_stringList(self):
        data = Table("zoo")
        var = data.domain["name"]

        fs = filter.FilterStringList
        filters = [
            ((["swan", "tuna", "wasp"], True), dict(rows=3)),
            ((["swan", "tuna", "wasp"], False), dict(rows=3)),
            ((["WoRm", "TOad", "vOLe"], True), dict(rows=0)),
            ((["WoRm", "TOad", "vOLe"], False), dict(rows=3)),
        ]

        for args, expected in filters:
            f = fs(var, *args)
            filtered_data = filter.Values([f])(data)
            self.assertEqual(len(filtered_data), expected["rows"],
                             "{} returned wrong number of rows".format(args))

    def test_table_dtypes(self):
        table = data.Table("iris")
        metas = np.hstack((table.metas, table.Y.reshape(len(table), 1)))
        attributes_metas = table.domain.metas + table.domain.class_vars
        domain_metas = data.Domain(table.domain.attributes[:2],
                                   table.domain.attributes[2:],
                                   attributes_metas)
        table_metas = data.Table(domain_metas, table.X[:, : 2], table.X[:, 2:],
                                 metas)
        new_table = data.Table.from_table(data.Domain(table_metas.domain.metas),
                                          table_metas)
        self.assertEqual(new_table.X.dtype, np.float64)
        new_table = data.Table.from_table(data.Domain((), table_metas.domain.metas),
                                          table_metas)
        self.assertEqual(new_table.Y.dtype, np.float64)
        new_table = data.Table.from_table(data.Domain((), (), table_metas.domain.metas),
                                          table_metas)
        self.assertEqual(new_table.metas.dtype, np.float64)

    def test_attributes(self):
        table = data.Table("iris")
        table.attributes = {1: "test"}
        table2 = table[:4]
        self.assertEqual(table2.attributes[1], "test")
        table2.attributes[1] = "modified"
        self.assertEqual(table.attributes[1], "modified")

    # TODO Test conjunctions and disjunctions of conditions

    def test_is_sparse(self):
        table = data.Table("iris")
        self.assertFalse(table.is_sparse())

        with table.unlocked():
            table.X = sp.csr_matrix(table.X)
            self.assertTrue(table.is_sparse())

    def test_repr_sparse_with_one_row(self):
        table = data.Table("iris")[:1]
        with table.unlocked_reference():
            table.X = sp.csr_matrix(table.X)
        repr(table)     # make sure repr does not crash

    def test_inf(self):
        a = np.array([[2, 0, 0, 0],
                      [0, np.nan, np.nan, 1],
                      [0, 0, np.inf, 0]])
        with self.assertWarns(Warning):
            tab = data.Table.from_numpy(None, a)
        self.assertEqual(tab.get_nan_frequency_attribute(), 3/12)


def column_sizes(table):
    return (len(table.domain.attributes),
            len(table.domain.class_vars),
            len(table.domain.metas))


class TableTests(unittest.TestCase):
    attributes = ["Feature %i" % i for i in range(10)]
    class_vars = ["Class %i" % i for i in range(1)]
    metas = ["Meta %i" % i for i in range(5)]
    nrows = 10
    row_indices = (1, 5, 7, 9)

    def setUp(self):
        self.data = np.random.random((self.nrows, len(self.attributes)))
        self.class_data = np.random.random((self.nrows, len(self.class_vars)))
        if len(self.class_vars) == 1:
            self.class_data = self.class_data.flatten()
        self.meta_data = np.random.randint(0, 5, (self.nrows, len(self.metas))
                                          ).astype(object)
        self.weight_data = np.random.random((self.nrows, 1))

    def mock_domain(self, with_classes=False, with_metas=False):
        attributes = self.attributes
        class_var = self.class_vars[0] if with_classes else None
        class_vars = self.class_vars if with_classes else []
        metas = self.metas if with_metas else []
        variables = attributes + class_vars
        return MagicMock(data.Domain,
                         attributes=attributes,
                         class_var=class_var,
                         class_vars=class_vars,
                         metas=metas,
                         variables=variables)

    def create_domain(self, attributes=(), classes=(), metas=()):
        attr_vars = [data.ContinuousVariable(name=a) if isinstance(a, str)
                     else a for a in attributes]
        class_vars = [data.ContinuousVariable(name=c) if isinstance(c, str)
                      else c for c in classes]
        meta_vars = [data.DiscreteVariable(name=m, values=map(str, range(5)))
                     if isinstance(m, str) else m for m in metas]

        domain = data.Domain(attr_vars, class_vars, meta_vars)
        return domain


class CreateEmptyTable(TableTests):
    def test_calling_new_with_no_parameters_constructs_a_new_instance(self):
        table = data.Table()
        self.assertIsInstance(table, data.Table)

    def test_table_has_file(self):
        table = data.Table()
        self.assertIsNone(table.__file__)


class CreateTableWithFilename(TableTests):
    filename = "data.tab"

    @patch("os.path.exists", Mock(return_value=True))
    def test_read_data_calls_reader(self):
        table_mock = Mock(data.Table)
        reader_instance = Mock(read=Mock(return_value=table_mock))
        reader_mock = Mock(return_value=reader_instance)

        with patch.dict(data.io.FileFormat.readers,
                        {'.xlsx': reader_mock}):
            table = data.Table.from_file("test.xlsx")

        reader_mock.assert_called_with("test.xlsx")
        reader_instance.read.assert_called_with()
        self.assertEqual(table, table_mock)

    @patch("os.path.exists", Mock(return_value=False))
    def test_raises_error_if_file_does_not_exist(self):
        with self.assertRaises(IOError):
            data.Table.from_file(self.filename)

    @patch("os.path.exists", Mock(return_value=True))
    def test_raises_error_if_file_has_unknown_extension(self):
        with self.assertRaises(IOError):
            data.Table.from_file("file.invalid_extension")

    @patch("Orange.data.table.Table.from_file")
    def test_calling_new_with_string_argument_calls_read_data(self, read_data):
        data.Table(self.filename)

        read_data.assert_called_with(self.filename)

    @patch("Orange.data.table.Table.from_file")
    def test_calling_new_with_keyword_argument_filename_calls_read_data(
            self, read_data):
        data.Table(self.filename)
        read_data.assert_called_with(self.filename)


class CreateTableWithUrl(TableTests):
    def test_url_no_scheme(self):

        class SkipRest(Exception):
            pass

        mock_urlopen = Mock(side_effect=SkipRest())
        url = 'www.foo.bar/xx.csv'

        with patch('Orange.data.io.UrlReader.urlopen', mock_urlopen):
            try:
                Table.from_url(url)
            except SkipRest:
                pass

        mock_urlopen.assert_called_once_with('http://' + url)

    class _MockUrlOpen(MagicMock):
        headers = {'content-disposition': 'attachment; filename="Something-FormResponses.tsv"; '
                                          'filename*=UTF-8''Something%20%28Responses%29.tsv'}
        def __enter__(self):
            return self

        def __exit__(self, *args, **kwargs):
            pass

        def read(self):
            return b'''\
a\tb\tc
1\t2\t3
2\t3\t4'''

    urlopen = _MockUrlOpen()

    @patch('Orange.data.io.urlopen', urlopen)
    def test_trimmed_urls(self):
        for url in ('https://docs.google.com/spreadsheets/d/ABCD/edit',
                    'https://www.dropbox.com/s/ABCD/filename.csv'):
            self._MockUrlOpen.url = url
            d = data.Table(url)
            request = self.urlopen.call_args[0][0]
            self.assertNotEqual(url, request.full_url)
            self.assertIn('Mozilla/5.0', request.headers.get('User-agent', ''))
            self.assertEqual(len(d), 2)
            self.assertEqual(d.name, 'Something-FormResponses')


class CreateTableWithDomain(TableTests):
    def test_creates_an_empty_table_with_given_domain(self):
        domain = self.mock_domain()
        table = data.Table.from_domain(domain)

        self.assertEqual(table.domain, domain)

    def test_creates_zero_filled_rows_in_X_if_domain_contains_attributes(self):
        domain = self.mock_domain()
        table = data.Table.from_domain(domain, self.nrows)

        self.assertEqual(table.X.shape, (self.nrows, len(domain.attributes)))
        self.assertFalse(table.X.any())

    def test_creates_zero_filled_rows_in_Y_if_domain_contains_class_vars(self):
        domain = self.mock_domain(with_classes=True)
        table = data.Table.from_domain(domain, self.nrows)

        if len(domain.class_vars) != 1:
            self.assertEqual(table.Y.shape,
                             (self.nrows, len(domain.class_vars)))
        else:
            self.assertEqual(table.Y.shape, (self.nrows,))
        self.assertFalse(table.Y.any())

    def test_creates_zero_filled_rows_in_metas_if_domain_contains_metas(self):
        domain = self.mock_domain(with_metas=True)
        table = data.Table.from_domain(domain, self.nrows)

        self.assertEqual(table.metas.shape, (self.nrows, len(domain.metas)))
        self.assertFalse(table.metas.any())

    def test_creates_weights_if_weights_are_true(self):
        domain = self.mock_domain()
        table = data.Table.from_domain(domain, self.nrows, True)

        self.assertEqual(table.W.shape, (self.nrows, ))

    def test_does_not_create_weights_if_weights_are_false(self):
        domain = self.mock_domain()
        table = data.Table.from_domain(domain, self.nrows, False)

        self.assertEqual(table.W.shape, (self.nrows, 0))

    @patch("Orange.data.table.Table.from_domain")
    def test_calling_new_with_domain_calls_new_from_domain(
            self, new_from_domain):
        domain = self.mock_domain()
        data.Table.from_domain(domain)

        new_from_domain.assert_called_with(domain)


class CreateTableWithData(TableTests):
    def test_creates_a_table_with_given_X(self):
        # from numpy
        table = data.Table.from_numpy(None, np.array(self.data))
        self.assertIsInstance(table.domain, data.Domain)
        np.testing.assert_almost_equal(table.X, self.data)

        # from list
        table = data.Table.from_numpy(None, list(self.data))
        self.assertIsInstance(table.domain, data.Domain)
        np.testing.assert_almost_equal(table.X, self.data)

        # from tuple
        table = data.Table.from_numpy(None, tuple(self.data))
        self.assertIsInstance(table.domain, data.Domain)
        np.testing.assert_almost_equal(table.X, self.data)

    def test_creates_a_table_from_domain_and_list(self):
        domain = data.Domain([data.DiscreteVariable(name="a", values="mf"),
                              data.ContinuousVariable(name="b")],
                             data.DiscreteVariable(name="y", values="abc"))
        table = data.Table.from_list(domain, [[0, 1, 2],
                                              [1, 2, "?"],
                                              ["m", 3, "a"],
                                              ["?", "?", "c"]])
        self.assertIs(table.domain, domain)
        np.testing.assert_almost_equal(
            table.X, np.array([[0, 1], [1, 2], [0, 3], [np.nan, np.nan]]))
        np.testing.assert_almost_equal(table.Y, np.array([2, np.nan, 0, 2]))

    def test_creates_a_table_from_domain_and_list_and_weights(self):
        domain = data.Domain([data.DiscreteVariable(name="a", values="mf"),
                              data.ContinuousVariable(name="b")],
                             data.DiscreteVariable(name="y", values="abc"))
        table = data.Table.from_list(domain, [[0, 1, 2],
                                              [1, 2, "?"],
                                              ["m", 3, "a"],
                                              ["?", "?", "c"]], [1, 2, 3, 4])
        self.assertIs(table.domain, domain)
        np.testing.assert_almost_equal(
            table.X, np.array([[0, 1], [1, 2], [0, 3], [np.nan, np.nan]]))
        np.testing.assert_almost_equal(table.Y, np.array([2, np.nan, 0, 2]))
        np.testing.assert_almost_equal(table.W, np.array([1, 2, 3, 4]))

    def test_creates_a_table_from_domain_and_list_and_metas(self):
        metas = [data.DiscreteVariable("Meta 1", values="XYZ"),
                 data.ContinuousVariable("Meta 2"),
                 data.StringVariable("Meta 3")]
        domain = data.Domain([data.DiscreteVariable(name="a", values="mf"),
                              data.ContinuousVariable(name="b")],
                             data.DiscreteVariable(name="y", values="abc"),
                             metas=metas)
        table = data.Table.from_list(domain, [[0, 1, 2, "X", 2, "bb"],
                                              [1, 2, "?", "Y", 1, "aa"],
                                              ["m", 3, "a", "Z", 3, "bb"],
                                              ["?", "?", "c", "X", 1, "aa"]])
        self.assertIs(table.domain, domain)
        np.testing.assert_almost_equal(
            table.X, np.array([[0, 1], [1, 2], [0, 3], [np.nan, np.nan]]))
        np.testing.assert_almost_equal(table.Y, np.array([2, np.nan, 0, 2]))
        np.testing.assert_array_equal(table.metas,
                                      np.array([[0, 2., "bb"],
                                                [1, 1., "aa"],
                                                [2, 3., "bb"],
                                                [0, 1., "aa"]],
                                               dtype=object))

    def test_creates_a_table_from_list_of_instances(self):
        table = data.Table('iris')
        new_table = data.Table.from_list(table.domain, [d for d in table])
        self.assertIs(table.domain, new_table.domain)
        np.testing.assert_almost_equal(table.X, new_table.X)
        np.testing.assert_almost_equal(table.Y, new_table.Y)
        np.testing.assert_almost_equal(table.W, new_table.W)
        self.assertEqual(table.domain, new_table.domain)
        np.testing.assert_array_equal(table.metas, new_table.metas)

    def test_creates_a_table_from_list_of_instances_with_metas(self):
        table = data.Table('zoo')
        new_table = data.Table.from_list(table.domain, [d for d in table])
        self.assertIs(table.domain, new_table.domain)
        np.testing.assert_almost_equal(table.X, new_table.X)
        np.testing.assert_almost_equal(table.Y, new_table.Y)
        np.testing.assert_almost_equal(table.W, new_table.W)
        self.assertEqual(table.domain, new_table.domain)
        np.testing.assert_array_equal(table.metas, new_table.metas)

    def test_creates_a_table_with_domain_and_given_X(self):
        domain = self.mock_domain()

        table = data.Table(domain, self.data)
        self.assertIsInstance(table.domain, data.Domain)
        self.assertEqual(table.domain, domain)
        np.testing.assert_almost_equal(table.X, self.data)

    def test_creates_a_table_with_given_X_and_Y(self):
        table = data.Table.from_numpy(None, self.data, self.class_data)

        self.assertIsInstance(table.domain, data.Domain)
        np.testing.assert_almost_equal(table.X, self.data)
        np.testing.assert_almost_equal(table.Y, self.class_data)

    def test_creates_a_table_with_given_X_Y_and_metas(self):
        table = data.Table.from_numpy(
            None, self.data, self.class_data, self.meta_data)

        self.assertIsInstance(table.domain, data.Domain)
        np.testing.assert_almost_equal(table.X, self.data)
        np.testing.assert_almost_equal(table.Y, self.class_data)
        np.testing.assert_almost_equal(table.metas, self.meta_data)

    def test_creates_a_discrete_class_if_Y_has_few_distinct_values(self):
        Y = np.array([float(np.random.randint(0, 2)) for i in self.data])
        table = data.Table.from_numpy(None, self.data, Y, self.meta_data)

        np.testing.assert_almost_equal(table.Y, Y)
        self.assertIsInstance(table.domain.class_vars[0],
                              data.DiscreteVariable)
        self.assertEqual(table.domain.class_vars[0].values, ("v1", "v2"))

    def test_creates_a_table_with_given_domain(self):
        domain = self.mock_domain()
        table = data.Table.from_numpy(domain, self.data)

        self.assertEqual(table.domain, domain)

    def test_sets_Y_if_given(self):
        domain = self.mock_domain(with_classes=True)
        table = data.Table.from_numpy(domain, self.data, self.class_data)

        np.testing.assert_almost_equal(table.Y, self.class_data)

    def test_sets_metas_if_given(self):
        domain = self.mock_domain(with_metas=True)
        table = data.Table.from_numpy(domain, self.data, metas=self.meta_data)

        np.testing.assert_almost_equal(table.metas, self.meta_data)

    def test_sets_weights_if_given(self):
        domain = self.mock_domain()
        table = data.Table.from_numpy(domain, self.data, W=self.weight_data)

        np.testing.assert_equal(table.W.shape, (len(self.data), ))
        np.testing.assert_almost_equal(table.W.flatten(),
                                       self.weight_data.flatten())

    def test_splits_X_and_Y_if_given_in_same_array(self):
        joined_data = np.column_stack((self.data, self.class_data))
        domain = self.mock_domain(with_classes=True)
        table = data.Table.from_numpy(domain, joined_data)

        np.testing.assert_almost_equal(table.X, self.data)
        np.testing.assert_almost_equal(table.Y, self.class_data)

    def test_initializes_Y_metas_and_W_if_not_given(self):
        domain = self.mock_domain()
        table = data.Table.from_numpy(domain, self.data)

        self.assertEqual(table.Y.shape, (self.nrows, len(domain.class_vars)))
        self.assertEqual(table.metas.shape, (self.nrows, len(domain.metas)))
        self.assertEqual(table.W.shape, (self.nrows, 0))

    def test_raises_error_if_columns_in_domain_and_data_do_not_match(self):
        domain = self.mock_domain(with_classes=True, with_metas=True)
        ones = np.zeros((self.nrows, 1))

        with self.assertRaises(ValueError):
            data_ = np.hstack((self.data, ones))
            data.Table.from_numpy(domain, data_, self.class_data,
                                  self.meta_data)

        with self.assertRaises(ValueError):
            classes_ = np.hstack((self.class_data, ones))
            data.Table.from_numpy(domain, self.data, classes_,
                                  self.meta_data)

        with self.assertRaises(ValueError):
            metas_ = np.hstack((self.meta_data, ones))
            data.Table.from_numpy(domain, self.data, self.class_data,
                                  metas_)

    def test_raises_error_if_lengths_of_data_do_not_match(self):
        domain = self.mock_domain(with_classes=True, with_metas=True)

        with self.assertRaises(ValueError):
            data_ = np.vstack((self.data, np.zeros((1, len(self.attributes)))))
            data.Table(domain, data_, self.class_data, self.meta_data)

        with self.assertRaises(ValueError):
            class_data_ = np.vstack((self.class_data,
                                     np.zeros((1, len(self.class_vars)))))
            data.Table(domain, self.data, class_data_, self.meta_data)

        with self.assertRaises(ValueError):
            meta_data_ = np.vstack((self.meta_data,
                                    np.zeros((1, len(self.metas)))))
            data.Table(domain, self.data, self.class_data, meta_data_)

    @patch("Orange.data.table.Table.from_numpy")
    def test_calling_new_with_domain_and_numpy_arrays_calls_new_from_numpy(
            self, new_from_numpy):
        domain = self.mock_domain()
        data.Table(domain, self.data)
        new_from_numpy.assert_called_with(domain, self.data)

        domain = self.mock_domain(with_classes=True)
        data.Table(domain, self.data, self.class_data)
        new_from_numpy.assert_called_with(domain, self.data, self.class_data)

        domain = self.mock_domain(with_classes=True, with_metas=True)
        data.Table(domain, self.data, self.class_data, self.meta_data)
        new_from_numpy.assert_called_with(
            domain, self.data, self.class_data, self.meta_data)

        data.Table(domain, self.data, self.class_data,
                   self.meta_data, self.weight_data)
        new_from_numpy.assert_called_with(domain, self.data, self.class_data,
                                          self.meta_data, self.weight_data)

    def test_from_numpy_reconstructable(self):
        def assert_equal(t1, t2):
            np.testing.assert_array_equal(t1.X, t2.X)
            np.testing.assert_array_equal(t1.Y, t2.Y)
            np.testing.assert_array_equal(t1.metas, t2.metas)
            np.testing.assert_array_equal(t1.W, t2.W)

        nullcol = np.empty((self.nrows, 0))
        domain = self.create_domain(self.attributes)
        table = data.Table(domain, self.data)

        table_1 = data.Table.from_numpy(
            domain, table.X, table.Y, table.metas, table.W)
        assert_equal(table, table_1)

        domain = self.create_domain(classes=self.class_vars)
        table = data.Table(domain, nullcol, self.class_data)

        table_1 = data.Table.from_numpy(
            domain, table.X, table.Y, table.metas, table.W)
        assert_equal(table, table_1)

        domain = self.create_domain(metas=self.metas)
        table = data.Table(domain, nullcol, nullcol, self.meta_data)

        table_1 = data.Table.from_numpy(
            domain, table.X, table.Y, table.metas, table.W)
        assert_equal(table, table_1)


class CreateTableWithDomainAndTable(TableTests):
    interesting_slices = [
        slice(0, 0),  # [0:0] - empty slice
        slice(1),  # [:1]  - only first element
        slice(1, None),  # [1:]  - all but first
        slice(-1, None),  # [-1:] - only last element
        slice(-1),  # [:-1] - all but last
        slice(None),  # [:]   - all elements
        slice(None, None, 2),  # [::2] - even elements
        slice(None, None, -1),  # [::-1]- all elements reversed
        slice(9, 5, -10),  # slice a big negative stride and thus 1 element
    ]

    row_indices = [1, 5, 6, 7]

    def setUp(self):
        super().setUp()
        self.domain = self.create_domain(
            self.attributes, self.class_vars, self.metas)
        self.table = data.Table(
            self.domain, self.data, self.class_data, self.meta_data)

    def test_creates_table_with_given_domain(self):
        new_table = data.Table.from_table(self.table.domain, self.table)

        self.assertIsInstance(new_table, data.Table)
        self.assertIsNot(self.table, new_table)
        self.assertEqual(new_table.domain, self.domain)

    def test_transform(self):
        class MyTableClass(data.Table):
            pass

        table = MyTableClass.from_table(self.table.domain, self.table)
        domain = table.domain
        attr = ContinuousVariable("x")
        new_domain = data.Domain(list(domain.attributes) + [attr], None)
        new_table = table.transform(new_domain)

        self.assertIsInstance(new_table, MyTableClass)
        self.assertIsNot(table, new_table)
        self.assertIs(new_table.domain, new_domain)

    def test_transform_same_domain(self):
        iris = data.Table("iris")
        new_domain = copy.copy(iris.domain)
        new_data = iris.transform(new_domain)
        self.assertIs(new_data.domain, new_domain)

    def test_can_copy_table(self):
        new_table = data.Table.from_table(self.domain, self.table)
        self.assert_table_with_filter_matches(new_table, self.table)

    def test_can_filter_rows_with_list(self):
        for indices in ([0], [1, 5, 6, 7]):
            new_table = data.Table.from_table(
                self.domain, self.table, row_indices=indices)
            self.assert_table_with_filter_matches(
                new_table, self.table, rows=indices)

    @patch.object(Table, "from_table_rows", wraps=Table.from_table_rows)
    def test_can_filter_row_with_slice_from_table_rows(self, from_table_rows):
        # calling from_table with the same domain will forward to from_table_rows
        for slice_ in self.interesting_slices:
            from_table_rows.reset_mock()
            new_table = data.Table.from_table(
                self.domain, self.table, row_indices=slice_)
            self.assert_table_with_filter_matches(
                new_table, self.table, rows=slice_)
            from_table_rows.assert_called()

    def test_can_filter_row_with_slice_from_table(self):
        # calling from_table with a domain copy will use indexing in from_table
        for slice_ in self.interesting_slices:
            new_table = data.Table.from_table(
                self.domain.copy(), self.table, row_indices=slice_)
            self.assert_table_with_filter_matches(
                new_table, self.table, rows=slice_)

    def test_can_use_attributes_as_new_columns(self):
        a, _, _ = column_sizes(self.table)
        order = np.random.permutation(a)
        new_attributes = [self.domain.attributes[i] for i in order]
        new_domain = self.create_domain(
            new_attributes[:2], new_attributes[2:4], new_attributes[4:])
        new_table = data.Table.from_table(new_domain, self.table)

        self.assert_table_with_filter_matches(
            new_table, self.table, xcols=order[:2], ycols=order[2:4], mcols=order[4:])

    def test_can_use_class_vars_as_new_columns(self):
        a, _, _ = column_sizes(self.table)
        order = np.random.permutation(range(a))
        cvs = [self.domain.attributes[i] for i in order[:2]]
        metas = [self.domain.attributes[i] for i in order[2:]]
        new_domain = self.create_domain(self.domain.class_vars,
                                        cvs,
                                        metas)
        new_table = data.Table.from_table(new_domain, self.table)

        self.assert_table_with_filter_matches(
            new_table, self.table, xcols=[a], ycols=order[:2], mcols=order[2:])

    def test_can_use_metas_as_new_columns(self):
        _, _, m = column_sizes(self.table)
        order = np.random.permutation(range(-m, 0))
        new_metas = [self.domain.metas[::-1][i] for i in order]
        new_domain = self.create_domain(new_metas[0:2], [new_metas[2]], new_metas[3:5])
        new_table = data.Table.from_table(new_domain, self.table)

        self.assert_table_with_filter_matches(
            new_table, self.table, xcols=order[0:2], ycols=order[2], mcols=order[3:5])

    def test_can_use_combination_of_all_as_new_columns(self):
        a, c, m = column_sizes(self.table)
        order = (list(range(a+c)) + list(range(-m+1, 0)))
        random. shuffle(order)
        vars_ = list(self.domain.variables) + list(self.domain.metas[::-1])
        atrs = [vars_[order[i]] for i in range(a)]
        cv = [vars_[order[i]] for i in range(a, a+c)]
        metas = [vars_[order[i]] for i in range(a+c, a+c+m-1)]

        new_domain = self.create_domain(atrs, cv, metas)
        new_table = data.Table.from_table(new_domain, self.table)
        self.assert_table_with_filter_matches(
            new_table, self.table, xcols=order[:a], ycols=order[a:a+c], mcols=order[a+c:])

    def test_creates_table_with_given_domain_and_row_filter(self):
        a, c, m = column_sizes(self.table)
        order = (list(range(a+c)) + list(range(-m+1, 0)))
        random.shuffle(order)
        vars_ = list(self.domain.variables) + list(self.domain.metas[::-1])
        atrs = [vars_[order[i]] for i in range(a)]
        cv = [vars_[order[i]] for i in range(a, a+c)]
        metas = [vars_[order[i]] for i in range(a+c, a+c+m-1)]

        new_domain = self.create_domain(atrs, cv, metas)
        new_table = data.Table.from_table(new_domain, self.table, [0])
        self.assert_table_with_filter_matches(
            new_table, self.table[:1], xcols=order[:a], ycols=order[a:a+c], mcols=order[a+c:])

        new_table = data.Table.from_table(new_domain, self.table, [2, 1, 0])
        self.assert_table_with_filter_matches(
            new_table, self.table[2::-1], xcols=order[:a], ycols=order[a:a+c], mcols=order[a+c:])

        new_table = data.Table.from_table(new_domain, self.table, [])
        self.assert_table_with_filter_matches(
            new_table, self.table[:0], xcols=order[:a], ycols=order[a:a+c], mcols=order[a+c:])

    def test_from_table_sparse_move_some_to_empty_metas(self):
        iris = data.Table("iris").to_sparse()
        new_domain = data.domain.Domain(
            iris.domain.attributes[:2], iris.domain.class_vars,
            iris.domain.attributes[2:], source=iris.domain)
        new_iris = iris.transform(new_domain)

        self.assertTrue(sp.issparse(new_iris.X))
        self.assertTrue(sp.issparse(new_iris.metas))
        self.assertEqual(new_iris.X.shape, (len(iris), 2))
        self.assertEqual(new_iris.metas.shape, (len(iris), 2))

        # move back
        back_iris = new_iris.transform(iris.domain)
        self.assertEqual(back_iris.domain, iris.domain)
        self.assertTrue(sp.issparse(back_iris.X))
        self.assertFalse(sp.issparse(back_iris.metas))
        self.assertEqual(back_iris.X.shape, iris.X.shape)
        self.assertEqual(back_iris.metas.shape, iris.metas.shape)

    def test_from_table_sparse_move_all_to_empty_metas(self):
        iris = data.Table("iris").to_sparse()
        new_domain = data.domain.Domain(
            [], iris.domain.class_vars, iris.domain.attributes,
            source=iris.domain)
        new_iris = iris.transform(new_domain)

        self.assertFalse(sp.issparse(new_iris.X))
        self.assertTrue(sp.issparse(new_iris.metas))
        self.assertEqual(new_iris.X.shape, (len(iris), 0))
        self.assertEqual(new_iris.metas.shape, (len(iris), 4))

        # move back
        back_iris = new_iris.transform(iris.domain)
        self.assertEqual(back_iris.domain, iris.domain)
        self.assertTrue(sp.issparse(back_iris.X))
        self.assertFalse(sp.issparse(back_iris.metas))
        self.assertEqual(back_iris.X.shape, iris.X.shape)
        self.assertEqual(back_iris.metas.shape, iris.metas.shape)

    def test_from_table_sparse_move_to_nonempty_metas(self):
        brown = data.Table("brown-selected").to_sparse()
        n_attr = len(brown.domain.attributes)
        n_metas = len(brown.domain.metas)
        new_domain = data.domain.Domain(
            brown.domain.attributes[:-10],
            brown.domain.class_vars,
            brown.domain.attributes[-10:] + brown.domain.metas,
            source=brown.domain)
        new_brown = data.Table.from_table(new_domain, brown)

        self.assertTrue(sp.issparse(new_brown.X))
        self.assertFalse(sp.issparse(new_brown.metas))
        self.assertEqual(new_brown.X.shape, (len(new_brown), n_attr-10))
        self.assertEqual(new_brown.metas.shape, (len(new_brown), n_metas+10))

        # move back
        back_brown = data.Table.from_table(brown.domain, new_brown)
        self.assertEqual(brown.domain, back_brown.domain)
        self.assertTrue(sp.issparse(back_brown.X))
        self.assertFalse(sp.issparse(back_brown.metas))
        self.assertEqual(back_brown.X.shape, brown.X.shape)
        self.assertEqual(back_brown.metas.shape, brown.metas.shape)

    def test_from_table_shared_compute_value(self):
        iris = data.Table("iris").to_sparse()
        d1 = Domain(
            [
                ContinuousVariable(
                    name=at.name,
                    compute_value=PreprocessSharedComputeValue(
                        i, None, PreprocessShared(iris.domain, None)
                    )
                )
                for i, at in enumerate(iris.domain.attributes)
            ]
        )

        new_table = Table.from_table(d1, iris)
        np.testing.assert_array_equal(new_table.X, iris.X.todense() * 2)

        new_table = Table.from_table(d1, iris, row_indices=[0, 1, 2])
        np.testing.assert_array_equal(new_table.X, iris.X.todense()[:3] * 2)

    def assert_table_with_filter_matches(
            self, new_table, old_table,
            rows=..., xcols=..., ycols=..., mcols=...):
        a, c, m = column_sizes(old_table)
        xcols = slice(a) if xcols is Ellipsis else xcols
        ycols = slice(a, a + c) if ycols is Ellipsis else ycols
        mcols = slice(None, -m - 1, -1) if mcols is Ellipsis else mcols

        # Indexing used by convert_domain uses positive indices for variables
        # and classes (classes come after attributes) and negative indices for
        # meta features. This is equivalent to ordinary indexing in a magic
        # table below.
        magic = np.hstack((old_table.X, old_table.Y[:, None],
                           old_table.metas[:, ::-1]))
        np.testing.assert_almost_equal(new_table.X, magic[rows, xcols])
        Y = magic[rows, ycols]
        if len(Y.shape) == 2 and Y.shape[1] == 1:
            Y = Y.flatten()
        np.testing.assert_almost_equal(new_table.Y, Y)
        np.testing.assert_almost_equal(new_table.metas, magic[rows, mcols])
        np.testing.assert_almost_equal(new_table.W, old_table.W[rows])


def isspecial(s):
    return isinstance(s, slice) or s is Ellipsis


def split_columns(indices, t):
    a, c, m = column_sizes(t)
    if indices is ...:
        return slice(a), slice(c), slice(m)
    elif isinstance(indices, slice):
        return indices, slice(0, 0), slice(0, 0)
    elif not isinstance(indices, list) and not isinstance(indices, tuple):
        indices = [indices]
    return (
        [t.domain.index(x)
         for x in indices if 0 <= t.domain.index(x) < a] or slice(0, 0),
        [t.domain.index(x) - a
         for x in indices if t.domain.index(x) >= a] or slice(0, 0),
        [-t.domain.index(x) - 1
         for x in indices if t.domain.index(x) < 0] or slice(0, 0))


def getname(variable):
    return variable.name


class TableIndexingTests(TableTests):
    def setUp(self):
        super().setUp()
        d = self.domain = \
            self.create_domain(self.attributes, self.class_vars, self.metas)
        t = self.table = \
            data.Table(self.domain, self.data, self.class_data, self.meta_data)
        self.magic_table = \
            np.column_stack((self.table.X, self.table.Y,
                             self.table.metas[:, ::-1]))

        self.rows = [0, -1]
        self.multiple_rows = [slice(0, 0), ..., slice(1, -1, -1)]
        a, c, m = column_sizes(t)
        columns = [0, a - 1, a, a + c - 1, -1, -m]
        self.columns = chain(columns,
                             map(lambda x: d[x], columns),
                             map(lambda x: d[x].name, columns))
        self.multiple_columns = chain(
            self.multiple_rows,
            [d.attributes, d.class_vars, d.metas, [0, a, -1]],
            [self.attributes, self.class_vars, self.metas],
            [self.attributes + self.class_vars + self.metas])

        # TODO: indexing with [[0,1], [0,1]] produces weird results
        # TODO: what should be the results of table[1, :]

    def test_can_select_a_single_value(self):
        for r in self.rows:
            for c in self.columns:
                value = self.table[r, c]
                self.assertAlmostEqual(
                    value, self.magic_table[r, self.domain.index(c)])

                value = self.table[r][c]
                self.assertAlmostEqual(
                    value, self.magic_table[r, self.domain.index(c)])

    def test_can_select_a_single_row(self):
        for r in self.rows:
            row = self.table[r]
            new_row = np.hstack(
                (self.data[r, :],
                 self.class_data[r, None]))
            np.testing.assert_almost_equal(
                np.array(list(row)), new_row)


    def test_can_select_a_subset_of_rows_and_columns(self):
        for r in self.rows:
            for c in self.multiple_columns:
                table = self.table[r, c]

                attr, cls, metas = split_columns(c, self.table)
                X = self.table.X[[r], attr]
                if X.ndim == 1:
                    X = X.reshape(-1, len(table.domain.attributes))
                np.testing.assert_almost_equal(table.X, X)
                Y = self.table.Y[:, None][[r], cls]
                if len(Y.shape) == 1 or Y.shape[1] == 1:
                    Y = Y.flatten()
                np.testing.assert_almost_equal(table.Y, Y)
                metas_ = self.table.metas[[r], metas]
                if metas_.ndim == 1:
                    metas_ = metas_.reshape(-1, len(table.domain.metas))
                np.testing.assert_almost_equal(table.metas, metas_)

        for r in self.multiple_rows:
            for c in chain(self.columns, self.multiple_rows):
                table = self.table[r, c]

                attr, cls, metas = split_columns(c, self.table)
                np.testing.assert_almost_equal(table.X, self.table.X[r, attr])
                Y = self.table.Y[:, None][r, cls]
                if len(Y.shape) > 1 and Y.shape[1] == 1:
                    Y = Y.flatten()
                np.testing.assert_almost_equal(table.Y, Y)
                np.testing.assert_almost_equal(table.metas,
                                               self.table.metas[r, metas])


    def test_optimize_indices(self):
        # ordinary conversion
        self.assertEqual(_optimize_indices([1, 2, 3], 4), slice(1, 4, 1))
        self.assertEqual(_optimize_indices([], 1), [])
        self.assertEqual(_optimize_indices([0, 2], 3), slice(0, 4, 2))

        # not slices
        np.testing.assert_equal(_optimize_indices([1, 2, 4], 5), [1, 2, 4])
        np.testing.assert_equal(_optimize_indices((1, 2, 4), 5), [1, 2, 4])

        # leave boolean arrays
        np.testing.assert_equal(_optimize_indices([True, False, True], 3), [True, False, True])

        # do not convert if step is negative
        np.testing.assert_equal(_optimize_indices([4, 2, 0], 5), [4, 2, 0])

        # try range
        np.testing.assert_equal(_optimize_indices([3, 4, 5], 5), [3, 4, 5])  # out of range
        self.assertEqual(_optimize_indices((3, 4, 5), 6), slice(3, 6, 1))

        # negative elements
        np.testing.assert_equal(_optimize_indices([-1, 0, 1], 5), [-1, 0, 1])

        # single element
        self.assertEqual(_optimize_indices([1], 2), slice(1, 2, 1))
        self.assertEqual(_optimize_indices([-2], 5), slice(-2, -3, -1))


class TableElementAssignmentTest(TableTests):
    def setUp(self):
        super().setUp()
        self.domain = \
            self.create_domain(self.attributes, self.class_vars, self.metas)
        self.table = \
            data.Table(self.domain, self.data, self.class_data, self.meta_data)

    def test_can_assign_values(self):
        with self.table.unlocked():
            self.table[0, 0] = 42.
        self.assertAlmostEqual(self.table.X[0, 0], 42.)

    def test_can_assign_values_to_classes(self):
        a, _, _ = column_sizes(self.table)
        with self.table.unlocked():
            self.table[0, a] = 42.
        self.assertAlmostEqual(self.table.Y[0], 42.)

    def test_can_assign_values_to_metas(self):
        with self.table.unlocked():
            self.table[0, -1] = 42.
        self.assertAlmostEqual(self.table.metas[0, 0], 42.)

    def test_can_assign_rows_to_rows(self):
        with self.table.unlocked():
            self.table[0] = self.table[1]
        np.testing.assert_almost_equal(
            self.table.X[0], self.table.X[1])
        np.testing.assert_almost_equal(
            self.table.Y[0], self.table.Y[1])
        np.testing.assert_almost_equal(
            self.table.metas[0], self.table.metas[1])

    def test_can_assign_lists(self):
        a, _, _ = column_sizes(self.table)
        new_example = [float(i)
                       for i in range(len(self.attributes + self.class_vars))]
        with self.table.unlocked():
            self.table[0] = new_example
        np.testing.assert_almost_equal(
            self.table.X[0], np.array(new_example[:a]))
        np.testing.assert_almost_equal(
            self.table.Y[0], np.array(new_example[a:]))

    def test_can_assign_np_array(self):
        a, _, _ = column_sizes(self.table)
        new_example = \
            np.array([float(i)
                      for i in range(len(self.attributes + self.class_vars))])
        with self.table.unlocked():
            self.table[0] = new_example
        np.testing.assert_almost_equal(self.table.X[0], new_example[:a])
        np.testing.assert_almost_equal(self.table.Y[0], new_example[a:])


class InterfaceTest(unittest.TestCase):
    """Basic tests each implementation of Table should pass."""

    features = (
        data.ContinuousVariable(name="Continuous Feature 1"),
        data.ContinuousVariable(name="Continuous Feature 2"),
        data.DiscreteVariable(name="Discrete Feature 1", values=("0", "1")),
        data.DiscreteVariable(name="Discrete Feature 2",
                              values=("value1", "value2")),
    )

    class_vars = (
        data.ContinuousVariable(name="Continuous Class"),
        data.DiscreteVariable(name="Discrete Class", values=("m", "f"))
    )

    feature_data = (
        (1, 0, 0, 0),
        (0, 1, 0, 0),
        (0, 0, 1, 0),
        (0, 0, 0, 1),
    )

    class_data = (
        (1, 0),
        (0, 1),
        (1, 0),
        (0, 1)
    )

    data = tuple(a + c for a, c in zip(feature_data, class_data))

    nrows = 4

    def setUp(self):
        self.domain = data.Domain(attributes=self.features,
                                  class_vars=self.class_vars)
        self.table = data.Table.from_numpy(
            self.domain,
            np.array(self.feature_data),
            np.array(self.class_data),
        )

    def test_len(self):
        self.assertEqual(len(self.table), self.nrows)

    def test_row_len(self):
        for i in range(self.nrows):
            self.assertEqual(len(self.table[i]), len(self.data[i]))

    def test_iteration(self):
        for row, expected_data in zip(self.table, self.data):
            self.assertEqual(tuple(row), expected_data)

    def test_row_indexing(self):
        for i in range(self.nrows):
            self.assertEqual(tuple(self.table[i]), self.data[i])

    def test_row_slicing(self):
        t = self.table[1:]
        self.assertEqual(len(t), self.nrows - 1)

    def test_value_indexing(self):
        for i in range(self.nrows):
            for j in range(len(self.table[i])):
                self.assertEqual(self.table[i, j], self.data[i][j])

    def test_row_assignment(self):
        new_value = 2.
        with self.table.unlocked():
            for i in range(self.nrows):
                new_row = [new_value] * len(self.data[i])
                self.table[i] = np.array(new_row)
                self.assertEqual(list(self.table[i]), new_row)

    def test_value_assignment(self):
        new_value = 0.
        with self.table.unlocked():
            for i in range(self.nrows):
                for j in range(len(self.table[i])):
                    self.table[i, j] = new_value
                    self.assertEqual(self.table[i, j], new_value)

    def test_subclasses(self):
        from pathlib import Path

        class _ExtendedTable(data.Table):
            pass

        data_file = _ExtendedTable('iris')
        data_url = _ExtendedTable.from_url(
            Path(os.path.dirname(__file__), 'datasets/test1.tab').as_uri())

        self.assertIsInstance(data_file, _ExtendedTable)
        self.assertIsInstance(data_url, _ExtendedTable)


class TestTableStats(TableTests):
    def test_get_nan_frequency(self):
        domain = self.create_domain(self.attributes, self.class_vars)
        table = data.Table(domain, self.data, self.class_data)
        self.assertEqual(table.get_nan_frequency_attribute(), 0)
        self.assertEqual(table.get_nan_frequency_class(), 0)

        with table.unlocked():
            table.X[1, 2] = table.X[4, 5] = np.nan
        self.assertEqual(table.get_nan_frequency_attribute(), 2 / table.X.size)
        self.assertEqual(table.get_nan_frequency_class(), 0)

        with table.unlocked():
            table.Y[3:6] = np.nan
        self.assertEqual(table.get_nan_frequency_attribute(), 2 / table.X.size)
        self.assertEqual(table.get_nan_frequency_class(), 3 / table.Y.size)

        with table.unlocked():
            table.X[1, 2] = table.X[4, 5] = 0
        self.assertEqual(table.get_nan_frequency_attribute(), 0)
        self.assertEqual(table.get_nan_frequency_class(), 3 / table.Y.size)

    def test_get_nan_frequency_empty_table(self):
        domain = self.create_domain(self.attributes, self.class_vars)
        table = data.Table.from_domain(domain)
        self.assertEqual(table.get_nan_frequency_attribute(), 0)
        self.assertEqual(table.get_nan_frequency_class(), 0)


class TestRowInstance(unittest.TestCase):
    def test_assignment(self):
        table = data.Table("zoo")
        inst = table[2]
        self.assertIsInstance(inst, data.RowInstance)

        with table.unlocked():
            inst[1] = 0
        self.assertEqual(table[2, 1], 0)
        with table.unlocked():
            inst[1] = 1
        self.assertEqual(table[2, 1], 1)

        with table.unlocked():
            inst.set_class("mammal")
        self.assertEqual(table[2, len(table.domain.attributes)], "mammal")
        with table.unlocked():
            inst.set_class("fish")
        self.assertEqual(table[2, len(table.domain.attributes)], "fish")

        with table.unlocked():
            inst[-1] = "Foo"
        self.assertEqual(table[2, -1], "Foo")

    def test_iteration_with_assignment(self):
        table = data.Table("iris")
        with table.unlocked():
            for i, row in enumerate(table):
                row[0] = i
        np.testing.assert_array_equal(table.X[:, 0], np.arange(len(table)))

    def test_sparse_assignment(self):
        X = np.eye(4)
        Y = X[2].copy()
        table = data.Table.from_numpy(None, X, Y)
        row = table[1]
        self.assertFalse(sp.issparse(row.sparse_x))
        self.assertEqual(row[0], 0)
        self.assertEqual(row[1], 1)

        with table.unlocked():
            table.X = sp.csr_matrix(table.X)
            table.Y = sp.csr_matrix(table.Y)
        sparse_row = table[1]
        self.assertTrue(sp.issparse(sparse_row.sparse_x))
        self.assertEqual(sparse_row[0], 0)
        self.assertEqual(sparse_row[1], 1)

        with table.unlocked():
            sparse_row[1] = 0
        self.assertEqual(sparse_row[1], 0)
        self.assertEqual(table.X[1, 1], 0)
        self.assertEqual(table[2][4], 1)

        with table.unlocked():
            table[2][4] = 0
        self.assertEqual(table[2][4], 0)


class TestTableTranspose(unittest.TestCase):
    def test_transpose_no_class(self):
        attrs = [ContinuousVariable("c1"), ContinuousVariable("c2")]
        table = Table(Domain(attrs), np.arange(8).reshape((4, 2)))

        att = [ContinuousVariable("Feature 1"), ContinuousVariable("Feature 2"),
               ContinuousVariable("Feature 3"), ContinuousVariable("Feature 4")]
        domain = Domain(att, metas=[StringVariable("Feature name")])
        result = Table(domain, np.arange(8).reshape((4, 2)).T,
                       metas=np.array(["c1", "c2"])[:, None])

        # transpose and compare
        self._compare_tables(result, Table.transpose(table))

        # transpose of transpose is original
        t = Table.transpose(Table.transpose(table), "Feature name")
        self._compare_tables(table, t)

        # original should not change
        self.assertDictEqual(table.domain.attributes[0].attributes, {})

    def test_transpose_discrete_class(self):
        attrs = [ContinuousVariable("c1"), ContinuousVariable("c2")]
        domain = Domain(attrs, [DiscreteVariable("cls", values=("a", "b"))])
        table = Table(domain, np.arange(8).reshape((4, 2)),
                     np.array([1, 1, 0, 0]))

        att = [ContinuousVariable("Feature 1"), ContinuousVariable("Feature 2"),
               ContinuousVariable("Feature 3"), ContinuousVariable("Feature 4")]
        att[0].attributes = {"cls": "b"}
        att[1].attributes = {"cls": "b"}
        att[2].attributes = {"cls": "a"}
        att[3].attributes = {"cls": "a"}
        domain = Domain(att, metas=[StringVariable("Feature name")])
        result = Table(domain, np.arange(8).reshape((4, 2)).T,
                       metas=np.array(["c1", "c2"])[:, None])

        # transpose and compare
        self._compare_tables(result, Table.transpose(table))

        # transpose of transpose is original
        t = Table.transpose(Table.transpose(table), "Feature name")
        self._compare_tables(table, t)

        # original should not change
        self.assertDictEqual(table.domain.attributes[0].attributes, {})

    def test_transpose_continuous_class(self):
        attrs = [ContinuousVariable("c1"), ContinuousVariable("c2")]
        domain = Domain(attrs, [ContinuousVariable("cls")])
        table = Table(domain, np.arange(8).reshape((4, 2)), np.arange(4, 0, -1))

        att = [ContinuousVariable("Feature 1"), ContinuousVariable("Feature 2"),
               ContinuousVariable("Feature 3"), ContinuousVariable("Feature 4")]
        att[0].attributes = {"cls": "4"}
        att[1].attributes = {"cls": "3"}
        att[2].attributes = {"cls": "2"}
        att[3].attributes = {"cls": "1"}
        domain = Domain(att, metas=[StringVariable("Feature name")])
        result = Table(domain, np.arange(8).reshape((4, 2)).T,
                       metas=np.array(["c1", "c2"])[:, None])

        # transpose and compare
        self._compare_tables(result, Table.transpose(table))

        # transpose of transpose
        t = Table.transpose(Table.transpose(table), "Feature name")
        self._compare_tables(table, t)

        # original should not change
        self.assertDictEqual(table.domain.attributes[0].attributes, {})

    def test_transpose_missing_class(self):
        attrs = [ContinuousVariable("c1"), ContinuousVariable("c2")]
        domain = Domain(attrs, [ContinuousVariable("cls")])
        table = Table(domain, np.arange(8).reshape((4, 2)),
                     np.array([np.nan, 3, 2, 1]))

        att = [ContinuousVariable("Feature 1"), ContinuousVariable("Feature 2"),
               ContinuousVariable("Feature 3"), ContinuousVariable("Feature 4")]
        att[1].attributes = {"cls": "3"}
        att[2].attributes = {"cls": "2"}
        att[3].attributes = {"cls": "1"}
        domain = Domain(att, metas=[StringVariable("Feature name")])
        result = Table(domain, np.arange(8).reshape((4, 2)).T,
                       metas=np.array(["c1", "c2"])[:, None])

        # transpose and compare
        self._compare_tables(result, Table.transpose(table))

        # transpose of transpose
        t = Table.transpose(Table.transpose(table), "Feature name")
        self._compare_tables(table, t)

        # original should not change
        self.assertDictEqual(table.domain.attributes[0].attributes, {})

    def test_transpose_multiple_class(self):
        attrs = [ContinuousVariable("c1"), ContinuousVariable("c2")]
        class_vars = [ContinuousVariable("cls1"), ContinuousVariable("cls2")]
        domain = Domain(attrs, class_vars)
        table = Table(domain, np.arange(8).reshape((4, 2)),
                      np.arange(8).reshape((4, 2)))

        att = [ContinuousVariable("Feature 1"), ContinuousVariable("Feature 2"),
               ContinuousVariable("Feature 3"), ContinuousVariable("Feature 4")]
        att[0].attributes = {"cls1": "0", "cls2": "1"}
        att[1].attributes = {"cls1": "2", "cls2": "3"}
        att[2].attributes = {"cls1": "4", "cls2": "5"}
        att[3].attributes = {"cls1": "6", "cls2": "7"}
        domain = Domain(att, metas=[StringVariable("Feature name")])
        result = Table(domain, np.arange(8).reshape((4, 2)).T,
                       metas=np.array(["c1", "c2"])[:, None])

        # transpose and compare
        self._compare_tables(result, Table.transpose(table))

        # transpose of transpose
        t = Table.transpose(Table.transpose(table), "Feature name")
        self._compare_tables(table, t)

        # original should not change
        self.assertDictEqual(table.domain.attributes[0].attributes, {})

    def test_transpose_metas(self):
        attrs = [ContinuousVariable("c1"), ContinuousVariable("c2")]
        metas = [StringVariable("m1")]
        domain = Domain(attrs, metas=metas)
        X = np.arange(8).reshape((4, 2))
        M = np.array(["aa", "bb", "cc", "dd"])[:, None]
        table = Table(domain, X, metas=M)

        att = [ContinuousVariable("Feature 1"), ContinuousVariable("Feature 2"),
               ContinuousVariable("Feature 3"), ContinuousVariable("Feature 4")]
        att[0].attributes = {"m1": "aa"}
        att[1].attributes = {"m1": "bb"}
        att[2].attributes = {"m1": "cc"}
        att[3].attributes = {"m1": "dd"}
        domain = Domain(att, metas=[StringVariable("Feature name")])
        result = Table(domain, np.arange(8).reshape((4, 2)).T,
                       metas=np.array(["c1", "c2"])[:, None])

        # transpose and compare
        self._compare_tables(result, Table.transpose(table))

        # transpose of transpose
        t = Table.transpose(Table.transpose(table), "Feature name")
        self._compare_tables(table, t)

        # original should not change
        self.assertDictEqual(table.domain.attributes[0].attributes, {})

    def test_transpose_discrete_metas(self):
        attrs = [ContinuousVariable("c1"), ContinuousVariable("c2")]
        metas = [DiscreteVariable("m1", values=("aa", "bb"))]
        domain = Domain(attrs, metas=metas)
        X = np.arange(8).reshape((4, 2))
        M = np.array([0, 1, 0, 1])[:, None]
        table = Table(domain, X, metas=M)

        att = [ContinuousVariable("Feature 1"), ContinuousVariable("Feature 2"),
               ContinuousVariable("Feature 3"), ContinuousVariable("Feature 4")]
        att[0].attributes = {"m1": "aa"}
        att[1].attributes = {"m1": "bb"}
        att[2].attributes = {"m1": "aa"}
        att[3].attributes = {"m1": "bb"}
        domain = Domain(att, metas=[StringVariable("Feature name")])
        result = Table(domain, np.arange(8).reshape((4, 2)).T,
                       metas=np.array(["c1", "c2"])[:, None])

        # transpose and compare
        self._compare_tables(result, Table.transpose(table))

        # transpose of transpose is original
        t = Table.transpose(Table.transpose(table), "Feature name")
        self._compare_tables(table, t)

        # original should not change
        self.assertDictEqual(table.domain.attributes[0].attributes, {})

    def test_transpose_continuous_metas(self):
        attrs = [ContinuousVariable("c1"), ContinuousVariable("c2")]
        metas = [ContinuousVariable("m1")]
        domain = Domain(attrs, metas=metas)
        X = np.arange(8).reshape((4, 2))
        M = np.array([0.0, 1.0, 0.0, 1.0])[:, None]
        table = Table(domain, X, metas=M)

        att = [ContinuousVariable("Feature 1"), ContinuousVariable("Feature 2"),
               ContinuousVariable("Feature 3"), ContinuousVariable("Feature 4")]
        att[0].attributes = {"m1": "0"}
        att[1].attributes = {"m1": "1"}
        att[2].attributes = {"m1": "0"}
        att[3].attributes = {"m1": "1"}
        domain = Domain(att, metas=[StringVariable("Feature name")])
        result = Table(domain, np.arange(8).reshape((4, 2)).T,
                       metas=np.array(["c1", "c2"])[:, None])

        # transpose and compare
        self._compare_tables(result, Table.transpose(table))

        # transpose of transpose is original
        t = Table.transpose(Table.transpose(table), "Feature name")
        self._compare_tables(table, t)

        # original should not change
        self.assertDictEqual(table.domain.attributes[0].attributes, {})

    def test_transpose_missing_metas(self):
        attrs = [ContinuousVariable("c1"), ContinuousVariable("c2")]
        metas = [StringVariable("m1")]
        domain = Domain(attrs, metas=metas)
        X = np.arange(8).reshape((4, 2))
        M = np.array(["aa", "bb", "", "dd"])[:, None]
        table = Table(domain, X, metas=M)

        att = [ContinuousVariable("Feature 1"), ContinuousVariable("Feature 2"),
               ContinuousVariable("Feature 3"), ContinuousVariable("Feature 4")]
        att[0].attributes = {"m1": "aa"}
        att[1].attributes = {"m1": "bb"}
        att[3].attributes = {"m1": "dd"}
        domain = Domain(att, metas=[StringVariable("Feature name")])
        result = Table(domain, np.arange(8).reshape((4, 2)).T,
                       metas=np.array(["c1", "c2"])[:, None])

        # transpose and compare
        self._compare_tables(result, Table.transpose(table))

        # transpose of transpose
        t = Table.transpose(Table.transpose(table), "Feature name")
        self._compare_tables(table, t)

        # original should not change
        self.assertDictEqual(table.domain.attributes[0].attributes, {})

    def test_transpose_multiple_metas(self):
        attrs = [ContinuousVariable("c1"), ContinuousVariable("c2")]
        metas = [StringVariable("m1"), StringVariable("m2")]
        domain = Domain(attrs, metas=metas)
        X = np.arange(8).reshape((4, 2))
        M = np.array([["aa", "aaa"], ["bb", "bbb"],
                      ["cc", "ccc"], ["dd", "ddd"]])
        table = Table(domain, X, metas=M)

        att = [ContinuousVariable("Feature 1"), ContinuousVariable("Feature 2"),
               ContinuousVariable("Feature 3"), ContinuousVariable("Feature 4")]
        att[0].attributes = {"m1": "aa", "m2": "aaa"}
        att[1].attributes = {"m1": "bb", "m2": "bbb"}
        att[2].attributes = {"m1": "cc", "m2": "ccc"}
        att[3].attributes = {"m1": "dd", "m2": "ddd"}
        domain = Domain(att, metas=[StringVariable("Feature name")])
        result = Table(domain, np.arange(8).reshape((4, 2)).T,
                       metas=np.array(["c1", "c2"])[:, None])

        # transpose and compare
        self._compare_tables(result, Table.transpose(table))

        # transpose of transpose
        t = Table.transpose(Table.transpose(table), "Feature name")
        self._compare_tables(table, t)

        # original should not change
        self.assertDictEqual(table.domain.attributes[0].attributes, {})

    def test_transpose_class_and_metas(self):
        attrs = [ContinuousVariable("c1"), ContinuousVariable("c2")]
        metas = [StringVariable("m1"), StringVariable("m2")]
        domain = Domain(attrs, [ContinuousVariable("cls")], metas)
        M = np.array([["aa", "aaa"], ["bb", "bbb"],
                      ["cc", "ccc"], ["dd", "ddd"]])
        table = Table(domain, np.arange(8).reshape((4, 2)), np.arange(1, 5), M)

        att = [ContinuousVariable("Feature 1"), ContinuousVariable("Feature 2"),
               ContinuousVariable("Feature 3"), ContinuousVariable("Feature 4")]
        att[0].attributes = {"cls": "1", "m1": "aa", "m2": "aaa"}
        att[1].attributes = {"cls": "2", "m1": "bb", "m2": "bbb"}
        att[2].attributes = {"cls": "3", "m1": "cc", "m2": "ccc"}
        att[3].attributes = {"cls": "4", "m1": "dd", "m2": "ddd"}
        domain = Domain(att, metas=[StringVariable("Feature name")])
        result = Table(domain, np.arange(8).reshape((4, 2)).T,
                       metas=np.array(["c1", "c2"])[:, None])

        # transpose and compare
        self._compare_tables(result, Table.transpose(table))

        # transpose of transpose
        t = Table.transpose(Table.transpose(table), "Feature name")
        self._compare_tables(table, t)

        # original should not change
        self.assertDictEqual(table.domain.attributes[0].attributes, {})

    def test_transpose_attributes_of_attributes_discrete(self):
        attrs = [ContinuousVariable("c1"), ContinuousVariable("c2")]
        attrs[0].attributes = {"attr1": "a", "attr2": "aa"}
        attrs[1].attributes = {"attr1": "b", "attr2": "bb"}
        domain = Domain(attrs)
        table = Table(domain, np.arange(8).reshape((4, 2)))

        att = [ContinuousVariable("Feature 1"), ContinuousVariable("Feature 2"),
               ContinuousVariable("Feature 3"), ContinuousVariable("Feature 4")]
        metas = [StringVariable("Feature name"),
                 DiscreteVariable("attr1", values=("a", "b")),
                 DiscreteVariable("attr2", values=("aa", "bb"))]
        domain = Domain(att, metas=metas)
        M = np.array([["c1", 0.0, 0.0], ["c2", 1.0, 1.0]], dtype=object)
        result = Table(domain, np.arange(8).reshape((4, 2)).T, metas=M)

        # transpose and compare
        self._compare_tables(result, Table.transpose(table))

        # transpose of transpose
        t = Table.transpose(Table.transpose(table), "Feature name")
        self._compare_tables(table, t)

        # original should not change
        self.assertDictEqual(table.domain.attributes[0].attributes,
                             {"attr1": "a", "attr2": "aa"})

    def test_transpose_attributes_of_attributes_continuous(self):
        attrs = [ContinuousVariable("c1"), ContinuousVariable("c2")]
        attrs[0].attributes = {"attr1": "1.1", "attr2": "1.3"}
        attrs[1].attributes = {"attr1": "2.2", "attr2": "2.3"}
        domain = Domain(attrs)
        table = Table(domain, np.arange(8).reshape((4, 2)))

        att = [ContinuousVariable("Feature 1"), ContinuousVariable("Feature 2"),
               ContinuousVariable("Feature 3"), ContinuousVariable("Feature 4")]
        metas = [StringVariable("Feature name"), ContinuousVariable("attr1"),
                 ContinuousVariable("attr2")]
        domain = Domain(att, metas=metas)
        result = Table(domain, np.arange(8).reshape((4, 2)).T,
                       metas=np.array([["c1", 1.1, 1.3],
                                       ["c2", 2.2, 2.3]], dtype=object))

        # transpose and compare
        self._compare_tables(result, Table.transpose(table))

        # transpose of transpose
        t = Table.transpose(Table.transpose(table), "Feature name")
        self._compare_tables(table, t)

        # original should not change
        self.assertDictEqual(table.domain.attributes[0].attributes,
                             {"attr1": "1.1", "attr2": "1.3"})

    def test_transpose_attributes_of_attributes_missings(self):
        attrs = [ContinuousVariable("c1"), ContinuousVariable("c2")]
        attrs[0].attributes = {"attr1": "a", "attr2": "aa"}
        attrs[1].attributes = {"attr1": "b"}
        domain = Domain(attrs)
        table = Table(domain, np.arange(8).reshape((4, 2)))

        att = [ContinuousVariable("Feature 1"), ContinuousVariable("Feature 2"),
               ContinuousVariable("Feature 3"), ContinuousVariable("Feature 4")]
        metas = [StringVariable("Feature name"),
                 DiscreteVariable("attr1", values=("a", "b")),
                 DiscreteVariable("attr2", values=("aa",))]
        domain = Domain(att, metas=metas)
        M = np.array([["c1", 0.0, 0.0], ["c2", 1.0, np.nan]], dtype=object)
        result = Table(domain, np.arange(8).reshape((4, 2)).T, metas=M)

        # transpose and compare
        self._compare_tables(result, Table.transpose(table))

        # transpose of transpose
        t = Table.transpose(Table.transpose(table), "Feature name")
        self._compare_tables(table, t)

        # original should not change
        self.assertDictEqual(table.domain.attributes[0].attributes,
                             {"attr1": "a", "attr2": "aa"})

    def test_transpose_class_metas_attributes(self):
        attrs = [ContinuousVariable("c1"), ContinuousVariable("c2")]
        attrs[0].attributes = {"attr1": "a1", "attr2": "aa1"}
        attrs[1].attributes = {"attr1": "b1", "attr2": "bb1"}
        metas = [StringVariable("m1"), StringVariable("m2")]
        domain = Domain(attrs, [ContinuousVariable("cls")], metas)
        M = np.array([["aa", "aaa"], ["bb", "bbb"],
                      ["cc", "ccc"], ["dd", "ddd"]])
        table = Table(domain, np.arange(8).reshape((4, 2)), np.arange(1, 5), M)

        att = [ContinuousVariable("Feature 1"), ContinuousVariable("Feature 2"),
               ContinuousVariable("Feature 3"), ContinuousVariable("Feature 4")]
        att[0].attributes = {"cls": "1", "m1": "aa", "m2": "aaa"}
        att[1].attributes = {"cls": "2", "m1": "bb", "m2": "bbb"}
        att[2].attributes = {"cls": "3", "m1": "cc", "m2": "ccc"}
        att[3].attributes = {"cls": "4", "m1": "dd", "m2": "ddd"}
        metas = [StringVariable("Feature name"),
                 DiscreteVariable("attr1", values=("a1", "b1")),
                 DiscreteVariable("attr2", values=("aa1", "bb1"))]
        domain = Domain(att, metas=metas)
        M = np.array([["c1", 0.0, 0.0], ["c2", 1.0, 1.0]], dtype=object)
        result = Table(domain, np.arange(8).reshape((4, 2)).T, metas=M)

        # transpose and compare
        self._compare_tables(result, Table.transpose(table))

        # transpose of transpose
        t = Table.transpose(Table.transpose(table), "Feature name")
        self._compare_tables(table, t)

        # original should not change
        self.assertDictEqual(table.domain.attributes[0].attributes,
                             {"attr1": "a1", "attr2": "aa1"})

    def test_transpose_duplicate_feature_names(self):
        table = Table("iris")
        domain = table.domain
        attrs, metas = domain.attributes[:3], domain.attributes[3:]
        table = table.transform(Domain(attrs, domain.class_vars, metas))
        transposed = Table.transpose(table, domain.attributes[3].name)
        names = [f.name for f in transposed.domain.attributes]
        self.assertEqual(len(names), len(set(names)))

    def test_transpose(self):
        zoo = Table("zoo")
        t1 = Table.transpose(zoo)
        t2 = Table.transpose(t1, "Feature name")
        t3 = Table.transpose(t2)
        self._compare_tables(zoo, t2)
        self._compare_tables(t1, t3)

    def test_transpose_callback(self):
        zoo = Table("zoo")
        cb = Mock()
        Table.transpose(zoo, progress_callback=cb)
        cb.assert_called()

    def test_transpose_no_class_remove_inst(self):
        attrs = [ContinuousVariable("c1"), ContinuousVariable("c2")]
        table = Table(Domain(attrs), np.arange(8).reshape((4, 2)))

        att = [ContinuousVariable("1"), ContinuousVariable("3"),
               ContinuousVariable("5"), ContinuousVariable("7")]
        domain = Domain(att, metas=[StringVariable("Feature name")])
        result = Table(domain, np.array([[0, 2, 4, 6]]),
                       metas=np.array([["c1"]]))

        # transpose and compare
        transposed = Table.transpose(table, "c2", remove_redundant_inst=True)
        self._compare_tables(result, transposed)

        # transpose of transpose is not equal to the original
        Table.transpose(transposed, "Feature name")
        Table.transpose(transposed, "3", remove_redundant_inst=True)

        # original should not change
        self.assertDictEqual(table.domain.attributes[0].attributes, {})

    def test_transpose_discrete_class_remove_inst(self):
        attrs = [ContinuousVariable("c1"), ContinuousVariable("c2")]
        domain = Domain(attrs, [DiscreteVariable("cls", values=("a", "b"))])
        table = Table(domain, np.arange(8).reshape((4, 2)),
                     np.array([1, 1, 0, 0]))

        att = [ContinuousVariable("1"), ContinuousVariable("3"),
               ContinuousVariable("5"), ContinuousVariable("7")]
        att[0].attributes = {"cls": "b"}
        att[1].attributes = {"cls": "b"}
        att[2].attributes = {"cls": "a"}
        att[3].attributes = {"cls": "a"}
        domain = Domain(att, metas=[StringVariable("Feature name")])
        result = Table(domain, np.array([[0, 2, 4, 6]]),
                       metas=np.array([["c1"]]))

        # transpose and compare
        transposed = Table.transpose(table, "c2", remove_redundant_inst=True)
        self._compare_tables(result, transposed)

        # transpose of transpose is not equal to the original
        Table.transpose(transposed, "Feature name")
        Table.transpose(transposed, "3", remove_redundant_inst=True)

        # original should not change
        self.assertDictEqual(table.domain.attributes[0].attributes, {})

    def test_transpose_continuous_class_remove_inst(self):
        attrs = [ContinuousVariable("c1"), ContinuousVariable("c2")]
        domain = Domain(attrs, [ContinuousVariable("cls")])
        table = Table(domain, np.arange(8).reshape((4, 2)), np.arange(4, 0, -1))

        att = [ContinuousVariable("1"), ContinuousVariable("3"),
               ContinuousVariable("5"), ContinuousVariable("7")]
        att[0].attributes = {"cls": "4"}
        att[1].attributes = {"cls": "3"}
        att[2].attributes = {"cls": "2"}
        att[3].attributes = {"cls": "1"}
        domain = Domain(att, metas=[StringVariable("Feature name")])
        result = Table(domain, np.array([[0, 2, 4, 6]]),
                       metas=np.array([["c1"]]))

        # transpose and compare
        transposed = Table.transpose(table, "c2", remove_redundant_inst=True)
        self._compare_tables(result, transposed)

        # transpose of transpose is not equal to the original
        Table.transpose(transposed, "Feature name")
        Table.transpose(transposed, "3", remove_redundant_inst=True)

        # original should not change
        self.assertDictEqual(table.domain.attributes[0].attributes, {})

    def test_transpose_missing_class_remove_inst(self):
        attrs = [ContinuousVariable("c1"), ContinuousVariable("c2")]
        domain = Domain(attrs, [ContinuousVariable("cls")])
        table = Table(domain, np.arange(8).reshape((4, 2)),
                     np.array([np.nan, 3, 2, 1]))

        att = [ContinuousVariable("1"), ContinuousVariable("3"),
               ContinuousVariable("5"), ContinuousVariable("7")]
        att[1].attributes = {"cls": "3"}
        att[2].attributes = {"cls": "2"}
        att[3].attributes = {"cls": "1"}
        domain = Domain(att, metas=[StringVariable("Feature name")])
        result = Table(domain, np.array([[0, 2, 4, 6]]),
                       metas=np.array([["c1"]]))

        # transpose and compare
        transposed = Table.transpose(table, "c2", remove_redundant_inst=True)
        self._compare_tables(result, transposed)

        # transpose of transpose is not equal to the original
        Table.transpose(transposed, "Feature name")
        Table.transpose(transposed, "3", remove_redundant_inst=True)

        # original should not change
        self.assertDictEqual(table.domain.attributes[0].attributes, {})

    def test_transpose_multiple_class_remove_inst(self):
        attrs = [ContinuousVariable("c1"), ContinuousVariable("c2")]
        class_vars = [ContinuousVariable("cls1"), ContinuousVariable("cls2")]
        domain = Domain(attrs, class_vars)
        table = Table(domain, np.arange(8).reshape((4, 2)),
                     np.arange(8).reshape((4, 2)))

        att = [ContinuousVariable("1"), ContinuousVariable("3"),
               ContinuousVariable("5"), ContinuousVariable("7")]
        att[0].attributes = {"cls1": "0", "cls2": "1"}
        att[1].attributes = {"cls1": "2", "cls2": "3"}
        att[2].attributes = {"cls1": "4", "cls2": "5"}
        att[3].attributes = {"cls1": "6", "cls2": "7"}
        domain = Domain(att, metas=[StringVariable("Feature name")])
        result = Table(domain, np.array([[0, 2, 4, 6]]),
                       metas=np.array([["c1"]]))

        # transpose and compare
        transposed = Table.transpose(table, "c2", remove_redundant_inst=True)
        self._compare_tables(result, transposed)

        # transpose of transpose is not equal to the original
        Table.transpose(transposed, "Feature name")
        Table.transpose(transposed, "3", remove_redundant_inst=True)

        # original should not change
        self.assertDictEqual(table.domain.attributes[0].attributes, {})

    def test_transpose_metas_remove_inst(self):
        attrs = [ContinuousVariable("c1"), ContinuousVariable("c2")]
        metas = [StringVariable("m1")]
        domain = Domain(attrs, metas=metas)
        X = np.arange(8).reshape((4, 2))
        M = np.array(["aa", "bb", "cc", "dd"])[:, None]
        table = Table(domain, X, metas=M)

        att = [ContinuousVariable("1"), ContinuousVariable("3"),
               ContinuousVariable("5"), ContinuousVariable("7")]
        att[0].attributes = {"m1": "aa"}
        att[1].attributes = {"m1": "bb"}
        att[2].attributes = {"m1": "cc"}
        att[3].attributes = {"m1": "dd"}
        domain = Domain(att, metas=[StringVariable("Feature name")])
        result = Table(domain, np.array([[0, 2, 4, 6]]),
                       metas=np.array([["c1"]]))

        # transpose and compare
        transposed = Table.transpose(table, "c2", remove_redundant_inst=True)
        self._compare_tables(result, transposed)

        # transpose of transpose is not equal to the original
        Table.transpose(transposed, "Feature name")
        Table.transpose(transposed, "3", remove_redundant_inst=True)

        # original should not change
        self.assertDictEqual(table.domain.attributes[0].attributes, {})

    def test_transpose_discrete_metas_remove_inst(self):
        attrs = [ContinuousVariable("c1"), ContinuousVariable("c2")]
        metas = [DiscreteVariable("m1", values=("aa", "bb"))]
        domain = Domain(attrs, metas=metas)
        X = np.arange(8).reshape((4, 2))
        M = np.array([0, 1, 0, 1])[:, None]
        table = Table(domain, X, metas=M)

        att = [ContinuousVariable("1"), ContinuousVariable("3"),
               ContinuousVariable("5"), ContinuousVariable("7")]
        att[0].attributes = {"m1": "aa"}
        att[1].attributes = {"m1": "bb"}
        att[2].attributes = {"m1": "aa"}
        att[3].attributes = {"m1": "bb"}
        domain = Domain(att, metas=[StringVariable("Feature name")])
        result = Table(domain, np.array([[0, 2, 4, 6]]),
                       metas=np.array([["c1"]]))

        # transpose and compare
        transposed = Table.transpose(table, "c2", remove_redundant_inst=True)
        self._compare_tables(result, transposed)

        # transpose of transpose is not equal to the original
        Table.transpose(transposed, "Feature name")
        Table.transpose(transposed, "3", remove_redundant_inst=True)

        # original should not change
        self.assertDictEqual(table.domain.attributes[0].attributes, {})

    def test_transpose_continuous_metas_remove_inst(self):
        attrs = [ContinuousVariable("c1"), ContinuousVariable("c2")]
        metas = [ContinuousVariable("m1")]
        domain = Domain(attrs, metas=metas)
        X = np.arange(8).reshape((4, 2))
        M = np.array([0.0, 1.0, 0.0, 1.0])[:, None]
        table = Table(domain, X, metas=M)

        att = [ContinuousVariable("1"), ContinuousVariable("3"),
               ContinuousVariable("5"), ContinuousVariable("7")]
        att[0].attributes = {"m1": "0"}
        att[1].attributes = {"m1": "1"}
        att[2].attributes = {"m1": "0"}
        att[3].attributes = {"m1": "1"}
        domain = Domain(att, metas=[StringVariable("Feature name")])
        result = Table(domain, np.array([[0, 2, 4, 6]]),
                       metas=np.array([["c1"]]))

        # transpose and compare
        transposed = Table.transpose(table, "c2", remove_redundant_inst=True)
        self._compare_tables(result, transposed)

        # transpose of transpose is not equal to the original
        Table.transpose(transposed, "Feature name")
        Table.transpose(transposed, "3", remove_redundant_inst=True)

        # original should not change
        self.assertDictEqual(table.domain.attributes[0].attributes, {})

    def test_transpose_missing_metas_remove_inst(self):
        attrs = [ContinuousVariable("c1"), ContinuousVariable("c2")]
        metas = [StringVariable("m1")]
        domain = Domain(attrs, metas=metas)
        X = np.arange(8).reshape((4, 2))
        M = np.array(["aa", "bb", "", "dd"])[:, None]
        table = Table(domain, X, metas=M)

        att = [ContinuousVariable("1"), ContinuousVariable("3"),
               ContinuousVariable("5"), ContinuousVariable("7")]
        att[0].attributes = {"m1": "aa"}
        att[1].attributes = {"m1": "bb"}
        att[3].attributes = {"m1": "dd"}
        domain = Domain(att, metas=[StringVariable("Feature name")])
        result = Table(domain, np.array([[0, 2, 4, 6]]),
                       metas=np.array([["c1"]]))

        # transpose and compare
        transposed = Table.transpose(table, "c2", remove_redundant_inst=True)
        self._compare_tables(result, transposed)

        # transpose of transpose is not equal to the original
        Table.transpose(transposed, "Feature name")
        Table.transpose(transposed, "3", remove_redundant_inst=True)

        # original should not change
        self.assertDictEqual(table.domain.attributes[0].attributes, {})

    def test_transpose_multiple_metas_remove_inst(self):
        attrs = [ContinuousVariable("c1"), ContinuousVariable("c2")]
        metas = [StringVariable("m1"), StringVariable("m2")]
        domain = Domain(attrs, metas=metas)
        X = np.arange(8).reshape((4, 2))
        M = np.array([["aa", "aaa"], ["bb", "bbb"],
                      ["cc", "ccc"], ["dd", "ddd"]])
        table = Table(domain, X, metas=M)

        att = [ContinuousVariable("1"), ContinuousVariable("3"),
               ContinuousVariable("5"), ContinuousVariable("7")]
        att[0].attributes = {"m1": "aa", "m2": "aaa"}
        att[1].attributes = {"m1": "bb", "m2": "bbb"}
        att[2].attributes = {"m1": "cc", "m2": "ccc"}
        att[3].attributes = {"m1": "dd", "m2": "ddd"}
        domain = Domain(att, metas=[StringVariable("Feature name")])
        result = Table(domain, np.array([[0, 2, 4, 6]]),
                       metas=np.array([["c1"]]))

        # transpose and compare
        transposed = Table.transpose(table, "c2", remove_redundant_inst=True)
        self._compare_tables(result, transposed)

        # transpose of transpose is not equal to the original
        Table.transpose(transposed, "Feature name")
        Table.transpose(transposed, "3", remove_redundant_inst=True)

        # original should not change
        self.assertDictEqual(table.domain.attributes[0].attributes, {})

    def test_transpose_class_and_metas_remove_inst(self):
        attrs = [ContinuousVariable("c1"), ContinuousVariable("c2")]
        metas = [StringVariable("m1"), StringVariable("m2")]
        domain = Domain(attrs, [ContinuousVariable("cls")], metas)
        M = np.array([["aa", "aaa"], ["bb", "bbb"],
                      ["cc", "ccc"], ["dd", "ddd"]])
        table = Table(domain, np.arange(8).reshape((4, 2)), np.arange(1, 5), M)

        att = [ContinuousVariable("1"), ContinuousVariable("3"),
               ContinuousVariable("5"), ContinuousVariable("7")]
        att[0].attributes = {"cls": "1", "m1": "aa", "m2": "aaa"}
        att[1].attributes = {"cls": "2", "m1": "bb", "m2": "bbb"}
        att[2].attributes = {"cls": "3", "m1": "cc", "m2": "ccc"}
        att[3].attributes = {"cls": "4", "m1": "dd", "m2": "ddd"}
        domain = Domain(att, metas=[StringVariable("Feature name")])
        result = Table(domain, np.array([[0, 2, 4, 6]]),
                       metas=np.array([["c1"]]))

        # transpose and compare
        transposed = Table.transpose(table, "c2", remove_redundant_inst=True)
        self._compare_tables(result, transposed)

        # transpose of transpose is not equal to the original
        Table.transpose(transposed, "Feature name")
        Table.transpose(transposed, "3", remove_redundant_inst=True)

        # original should not change
        self.assertDictEqual(table.domain.attributes[0].attributes, {})

    def test_transpose_attributes_of_attributes_discrete_remove_inst(self):
        attrs = [ContinuousVariable("c1"), ContinuousVariable("c2")]
        attrs[0].attributes = {"attr1": "a", "attr2": "aa"}
        attrs[1].attributes = {"attr1": "b", "attr2": "bb"}
        domain = Domain(attrs)
        table = Table(domain, np.arange(8).reshape((4, 2)))

        att = [ContinuousVariable("1"), ContinuousVariable("3"),
               ContinuousVariable("5"), ContinuousVariable("7")]
        metas = [StringVariable("Feature name"),
                 DiscreteVariable("attr1", values=("a", "b")),
                 DiscreteVariable("attr2", values=("aa", "bb"))]
        domain = Domain(att, metas=metas)
        result = Table(domain, np.array([[0, 2, 4, 6]]),
                       metas=np.array([["c1", 0.0, 0.0]], dtype=object))

        # transpose and compare
        transposed = Table.transpose(table, "c2", remove_redundant_inst=True)
        self._compare_tables(result, transposed)

        # transpose of transpose is not equal to the original
        Table.transpose(transposed, "Feature name")
        Table.transpose(transposed, "3", remove_redundant_inst=True)

        # original should not change
        self.assertDictEqual(table.domain.attributes[0].attributes,
                             {"attr1": "a", "attr2": "aa"})

    def test_transpose_attributes_of_attributes_continuous_remove_inst(self):
        attrs = [ContinuousVariable("c1"), ContinuousVariable("c2")]
        attrs[0].attributes = {"attr1": "1.1", "attr2": "1.3"}
        attrs[1].attributes = {"attr1": "2.2", "attr2": "2.3"}
        domain = Domain(attrs)
        table = Table(domain, np.arange(8).reshape((4, 2)))

        att = [ContinuousVariable("1"), ContinuousVariable("3"),
               ContinuousVariable("5"), ContinuousVariable("7")]
        metas = [StringVariable("Feature name"), ContinuousVariable("attr1"),
                 ContinuousVariable("attr2")]
        domain = Domain(att, metas=metas)
        result = Table(domain, np.array([[0, 2, 4, 6]]),
                       metas=np.array([["c1", 1.1, 1.3]], dtype=object))

        # transpose and compare
        transposed = Table.transpose(table, "c2", remove_redundant_inst=True)
        self._compare_tables(result, transposed)

        # transpose of transpose is not equal to the original
        Table.transpose(transposed, "Feature name")
        Table.transpose(transposed, "3", remove_redundant_inst=True)

        # original should not change
        self.assertDictEqual(table.domain.attributes[0].attributes,
                             {"attr1": "1.1", "attr2": "1.3"})

    def test_transpose_attributes_of_attributes_missings_remove_inst(self):
        attrs = [ContinuousVariable("c1"), ContinuousVariable("c2")]
        attrs[0].attributes = {"attr1": "a", "attr2": "aa"}
        attrs[1].attributes = {"attr1": "b"}
        domain = Domain(attrs)
        table = Table(domain, np.arange(8).reshape((4, 2)))

        att = [ContinuousVariable("0"), ContinuousVariable("2"),
               ContinuousVariable("4"), ContinuousVariable("6")]
        metas = [StringVariable("Feature name"),
                 DiscreteVariable("attr1", values=("b",))]
        domain = Domain(att, metas=metas)
        result = Table(domain, np.array([[1, 3, 5, 7]]),
                       metas=np.array([["c2", 0.0]], dtype=object))

        # transpose and compare
        transposed = Table.transpose(table, "c1", remove_redundant_inst=True)
        self._compare_tables(result, transposed)

        # transpose of transpose is not equal to the original
        Table.transpose(transposed, "Feature name")
        Table.transpose(transposed, "2", remove_redundant_inst=True)

        # original should not change
        self.assertDictEqual(table.domain.attributes[0].attributes,
                             {"attr1": "a", "attr2": "aa"})

    def test_transpose_class_metas_attributes_remove_inst(self):
        attrs = [ContinuousVariable("c1"), ContinuousVariable("c2")]
        attrs[0].attributes = {"attr1": "a1", "attr2": "aa1"}
        attrs[1].attributes = {"attr1": "b1", "attr2": "bb1"}
        metas = [StringVariable("m1"), StringVariable("m2")]
        domain = Domain(attrs, [ContinuousVariable("cls")], metas)
        M = np.array([["aa", "aaa"], ["bb", "bbb"],
                      ["cc", "ccc"], ["dd", "ddd"]])
        table = Table(domain, np.arange(8).reshape((4, 2)), np.arange(1, 5), M)

        att = [ContinuousVariable("1"), ContinuousVariable("3"),
               ContinuousVariable("5"), ContinuousVariable("7")]
        att[0].attributes = {"cls": "1", "m1": "aa", "m2": "aaa"}
        att[1].attributes = {"cls": "2", "m1": "bb", "m2": "bbb"}
        att[2].attributes = {"cls": "3", "m1": "cc", "m2": "ccc"}
        att[3].attributes = {"cls": "4", "m1": "dd", "m2": "ddd"}
        metas = [StringVariable("Feature name"),
                 DiscreteVariable("attr1", values=("a1", "b1")),
                 DiscreteVariable("attr2", values=("aa1", "bb1"))]
        domain = Domain(att, metas=metas)
        result = Table(domain, np.array([[0, 2, 4, 6]]),
                       metas=np.array([["c1", 0.0, 0.0]], dtype=object))

        # transpose and compare
        transposed = Table.transpose(table, "c2", remove_redundant_inst=True)
        self._compare_tables(result, transposed)

        # transpose of transpose is not equal to the original
        Table.transpose(transposed, "Feature name")
        Table.transpose(transposed, "3", remove_redundant_inst=True)

        # original should not change
        self.assertDictEqual(table.domain.attributes[0].attributes,
                             {"attr1": "a1", "attr2": "aa1"})

    def _compare_tables(self, table1, table2):
        self.assertEqual(table1.n_rows, table2.n_rows)
        np.testing.assert_array_equal(table1.X, table2.X)
        np.testing.assert_array_equal(table1.Y, table2.Y)
        np.testing.assert_array_equal(table1.metas.astype(str),
                                      table2.metas.astype(str))
        np.testing.assert_array_equal(table1.W, table2.W)

        self.assertEqual([(type(x), x.name, x.attributes)
                          for x in table1.domain.attributes],
                         [(type(x), x.name, x.attributes)
                          for x in table2.domain.attributes])
        self.assertEqual([(type(x), x.name, x.attributes)
                          for x in table1.domain.class_vars],
                         [(type(x), x.name, x.attributes)
                          for x in table2.domain.class_vars])
        self.assertEqual([(type(x), x.name, x.attributes)
                          for x in table1.domain.metas],
                         [(type(x), x.name, x.attributes)
                          for x in table2.domain.metas])


class SparseCV:
    def __call__(self, data):
        return sp.csr_matrix((len(data), 1))


class TestTableSparseDense(unittest.TestCase):
    def setUp(self):
        self.iris = Table('iris')

    def test_sparse_dense_transformation(self):
        iris = Table('iris')
        iris_sparse = iris.to_sparse(sparse_attributes=True)
        self.assertTrue(sp.issparse(iris_sparse.X))
        self.assertFalse(sp.issparse(iris_sparse.Y))
        self.assertFalse(sp.issparse(iris_sparse.metas))

        iris_sparse = iris.to_sparse(sparse_attributes=True, sparse_class=True)
        self.assertTrue(sp.issparse(iris_sparse.X))
        self.assertFalse(sp.issparse(iris_sparse.Y))
        self.assertFalse(sp.issparse(iris_sparse.metas))

        dense_iris = iris_sparse.to_dense()
        self.assertFalse(sp.issparse(dense_iris.X))
        self.assertFalse(sp.issparse(dense_iris.Y))
        self.assertFalse(sp.issparse(dense_iris.metas))

    def test_from_table_add_one_sparse_column(self):
        # add one sparse feature, should remain dense
        domain = self.iris.domain.copy()
        domain.attributes += (
            ContinuousVariable('S1', compute_value=SparseCV(), sparse=True),
        )
        d = self.iris.transform(domain)
        self.assertFalse(sp.issparse(d.X))

        # try with indices that are not Ellipsis
        d = Table.from_table(domain, self.iris, row_indices=[0, 1, 2])
        np.testing.assert_array_equal(
            d.X,
            [[5.1, 3.5, 1.4, 0.2, 0],
             [4.9, 3.0, 1.4, 0.2, 0],
             [4.7, 3.2, 1.3, 0.2, 0]]
        )
        self.assertFalse(sp.issparse(d.X))

    def test_from_table_add_lots_of_sparse_columns(self):
        n_attrs = len(self.iris.domain.attributes)

        # add 2*n_attrs+1 sparse feature, should became sparse
        domain = self.iris.domain.copy()
        domain.attributes += tuple(
            ContinuousVariable('S' + str(i), compute_value=SparseCV(), sparse=True)
            for i in range(2*n_attrs + 1)
        )
        d = self.iris.transform(domain)
        self.assertTrue(sp.issparse(d.X))

    def test_from_table_replace_attrs_with_sparse(self):
        # replace attrs with a sparse feature, should became sparse
        domain = self.iris.domain.copy()
        domain.attributes = (
            ContinuousVariable('S1', compute_value=SparseCV(), sparse=True),
        )
        d = self.iris.transform(domain)
        self.assertTrue(sp.issparse(d.X))

    def test_from_table_sparse_metas(self):
        # replace metas with a sparse feature, should became sparse
        domain = self.iris.domain.copy()
        domain._metas = (
            ContinuousVariable('S1', compute_value=SparseCV(), sparse=True),
        )
        d = self.iris.transform(domain)
        self.assertTrue(sp.issparse(d.metas))

    def test_from_table_sparse_metas_with_strings(self):
        # replace metas with text and 100 sparse features, should be dense
        domain = self.iris.domain.copy()
        domain._metas = (StringVariable('text'),) + tuple(
            ContinuousVariable('S' + str(i), compute_value=SparseCV(), sparse=True)
            for i in range(100)
        )
        d = self.iris.transform(domain)
        self.assertFalse(sp.issparse(d.metas))


class ConcurrencyTests(unittest.TestCase):

    def test_from_table_non_blocking(self):
        iris = Table("iris")[:10]

        def slow_compute_value(d):
            sleep(0.1)
            return d.X[:, 0]

        ndom = Domain([ContinuousVariable("a", compute_value=slow_compute_value)])

        def run_from_table():
            Table.from_table(ndom, iris)

        start = time()

        threads = []
        for _ in range(10):
            thread = Thread(target=run_from_table)
            thread.start()
            threads.append(thread)
        for t in threads:
            t.join()

        # if from_table was blocking these threads would need at least 0.1*10s
        duration = time() - start
        self.assertLess(duration, 0.5)


class PreprocessComputeValue:

    def __init__(self, domain, callback):
        self.domain = domain
        self.callback = callback

    def __call__(self, data_):
        if self.callback:
            self.callback(data_)
        transformed = data_.transform(self.domain)
        return transformed.X[:, 0] * 2


class PreprocessShared:

    def __init__(self, domain, callback):
        self.domain = domain
        self.callback = callback

    def __call__(self, data_):
        if self.callback:
            self.callback(data_)
        transformed = data_.transform(self.domain)
        return transformed.X * 2


class PreprocessSharedComputeValue(SharedComputeValue):

    def __init__(self, col, callback, shared):
        super().__init__(compute_shared=shared)
        self.col = col
        self.callback = callback

    # pylint: disable=arguments-differ
    def compute(self, data_, shared_data):
        if self.callback:
            self.callback(data_)
        return shared_data[:, self.col]


def preprocess_domain_single(domain, callback):
    """ Preprocess domain with single-source compute values.
    """
    return Domain([
        ContinuousVariable(name=at.name,
                           compute_value=PreprocessComputeValue(Domain([at]), callback))
        for at in domain.attributes])


def preprocess_domain_shared(domain, callback, callback_shared):
    """ Preprocess domain with shared compute values.
    """
    shared = PreprocessShared(domain, callback_shared)
    return Domain([
        ContinuousVariable(name=at.name,
                           compute_value=PreprocessSharedComputeValue(i, callback, shared))
        for i, at in enumerate(domain.attributes)])


def preprocess_domain_single_stupid(domain, callback):
    """ Preprocess domain with single-source compute values with stupid
    implementation: before applying it, instead of transforming just one column
    into the input domain, do a needless transform of the whole domain.
    """
    return Domain([
        ContinuousVariable(name=at.name,
                           compute_value=PreprocessComputeValue(domain, callback))
        for at in domain.attributes])


class EfficientTransformTests(unittest.TestCase):

    def setUp(self):
        self.iris = Table("iris")[:10]

    def test_simple(self):
        call_cv = Mock()
        d1 = preprocess_domain_single(self.iris.domain, call_cv)
        t = self.iris.transform(d1)
        self.assertEqual(4, call_cv.call_count)
        np.testing.assert_equal(t.X, self.iris.X * 2)

    def test_shared(self):
        call_cv = Mock()
        call_shared = Mock()
        d1 = preprocess_domain_shared(self.iris.domain, call_cv, call_shared)
        t = self.iris.transform(d1)
        self.assertEqual(4, call_cv.call_count)
        self.assertEqual(1, call_shared.call_count)
        np.testing.assert_equal(t.X, self.iris.X * 2)

    def test_simple_simple_shared(self):
        call_cv = Mock()
        d1 = preprocess_domain_single(self.iris.domain, call_cv)
        d2 = preprocess_domain_single(d1, call_cv)
        call_shared = Mock()
        d3 = preprocess_domain_shared(d2, call_cv, call_shared)
        t = self.iris.transform(d3)
        self.assertEqual(1, call_shared.call_count)
        self.assertEqual(12, call_cv.call_count)
        np.testing.assert_equal(t.X, self.iris.X * 2**3)

    def test_simple_simple_shared_simple(self):
        call_cv = Mock()
        d1 = preprocess_domain_single(self.iris.domain, call_cv)
        d2 = preprocess_domain_single(d1, call_cv)
        call_shared = Mock()
        d3 = preprocess_domain_shared(d2, call_cv, call_shared)
        d4 = preprocess_domain_single(d3, call_cv)
        t = self.iris.transform(d4)
        self.assertEqual(1, call_shared.call_count)
        self.assertEqual(16, call_cv.call_count)
        np.testing.assert_equal(t.X, self.iris.X * 2**4)

    def test_simple_simple_shared_simple_shared_simple(self):
        call_cv = Mock()
        d1 = preprocess_domain_single(self.iris.domain, call_cv)
        d2 = preprocess_domain_single(d1, call_cv)
        call_shared = Mock()
        d3 = preprocess_domain_shared(d2, call_cv, call_shared)
        d4 = preprocess_domain_single(d3, call_cv)
        d5 = preprocess_domain_shared(d4, call_cv, call_shared)
        d6 = preprocess_domain_single(d5, call_cv)
        t = self.iris.transform(d6)
        self.assertEqual(2, call_shared.call_count)
        self.assertEqual(24, call_cv.call_count)
        np.testing.assert_equal(t.X, self.iris.X * 2**6)

    def test_simple_simple_stupid(self):
        call_cv = Mock()
        d1 = preprocess_domain_single_stupid(self.iris.domain, call_cv)
        d2 = preprocess_domain_single_stupid(d1, call_cv)
        t = self.iris.transform(d2)
        self.assertEqual(8, call_cv.call_count)
        np.testing.assert_equal(t.X[:, 0], self.iris.X[:, 0] * 4)


if __name__ == "__main__":
    unittest.main()
