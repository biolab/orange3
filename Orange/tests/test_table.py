# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring

import os
import unittest
from itertools import chain
from math import isnan
import random

from unittest.mock import Mock, MagicMock, patch
import scipy.sparse as sp
import numpy as np

from Orange import data
from Orange.data import (filter, Unknown, Variable, Table, DiscreteVariable,
                         ContinuousVariable, Domain, StringVariable)
from Orange.tests import test_dirname, assert_array_nanequal


class TableTestCase(unittest.TestCase):
    def setUp(self):
        Variable._clear_all_caches()
        data.table.dataset_dirs.append(test_dirname())

    def tearDown(self):
        data.table.dataset_dirs.remove(test_dirname())

    def test_indexing_class(self):
        d = data.Table("test1")
        self.assertEqual([e.get_class() for e in d], ["t", "t", "f"])
        cind = len(d.domain) - 1
        self.assertEqual([e[cind] for e in d], ["t", "t", "f"])
        self.assertEqual([e["d"] for e in d], ["t", "t", "f"])
        cvar = d.domain.class_var
        self.assertEqual([e[cvar] for e in d], ["t", "t", "f"])

    def test_filename(self):
        dir = data.table.get_sample_datasets_dir()
        d = data.Table("iris")
        self.assertEqual(d.__file__, os.path.join(dir, "iris.tab"))

        d = data.Table("test2.tab")
        self.assertTrue(d.__file__.endswith("test2.tab"))  # platform dependent

    def test_indexing(self):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d = data.Table("test2")

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
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d = data.Table("test2")
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
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d = data.Table("test2")

            # meta
            vara = d.domain["a"]
            metaa = d.domain.index("a")

            self.assertEqual(d[0, "a"], "A")
            d[0, "a"] = "B"
            self.assertEqual(d[0, "a"], "B")
            d[0]["a"] = "A"
            self.assertEqual(d[0, "a"], "A")

            d[0, vara] = "B"
            self.assertEqual(d[0, "a"], "B")
            d[0][vara] = "A"
            self.assertEqual(d[0, "a"], "A")

            d[0, metaa] = "B"
            self.assertEqual(d[0, "a"], "B")
            d[0][metaa] = "A"
            self.assertEqual(d[0, "a"], "A")

            d[0, np.int_(metaa)] = "B"
            self.assertEqual(d[0, "a"], "B")
            d[0][np.int_(metaa)] = "A"
            self.assertEqual(d[0, "a"], "A")

            # regular
            varb = d.domain["b"]

            self.assertEqual(d[0, "b"], 0)
            d[0, "b"] = 42
            self.assertEqual(d[0, "b"], 42)
            d[0]["b"] = 0
            self.assertEqual(d[0, "b"], 0)

            d[0, varb] = 42
            self.assertEqual(d[0, "b"], 42)
            d[0][varb] = 0
            self.assertEqual(d[0, "b"], 0)

            d[0, 0] = 42
            self.assertEqual(d[0, "b"], 42)
            d[0][0] = 0
            self.assertEqual(d[0, "b"], 0)

            d[0, np.int_(0)] = 42
            self.assertEqual(d[0, "b"], 42)
            d[0][np.int_(0)] = 0
            self.assertEqual(d[0, "b"], 0)

    def test_indexing_del_example(self):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d = data.Table("test2")
            initlen = len(d)

            # remove first
            d[4, "e"] = "4ex"
            self.assertEqual(d[4, "e"], "4ex")
            del d[0]
            self.assertEqual(len(d), initlen - 1)
            self.assertEqual(d[3, "e"], "4ex")

            # remove middle
            del d[2]
            self.assertEqual(len(d), initlen - 2)
            self.assertEqual(d[2, "e"], "4ex")

            # remove middle
            del d[4]
            self.assertEqual(len(d), initlen - 3)
            self.assertEqual(d[2, "e"], "4ex")

            # remove last
            d[-1, "e"] = "was last"
            del d[-1]
            self.assertEqual(len(d), initlen - 4)
            self.assertEqual(d[2, "e"], "4ex")
            self.assertNotEqual(d[-1, "e"], "was last")

            # remove one before last
            d[-1, "e"] = "was last"
            del d[-2]
            self.assertEqual(len(d), initlen - 5)
            self.assertEqual(d[2, "e"], "4ex")
            self.assertEqual(d[-1, "e"], "was last")

            d[np.int_(2), "e"] = "2ex"
            del d[np.int_(2)]
            self.assertEqual(len(d), initlen - 6)
            self.assertNotEqual(d[2, "e"], "2ex")

            with self.assertRaises(IndexError):
                del d[100]
            self.assertEqual(len(d), initlen - 6)

            with self.assertRaises(IndexError):
                del d[-100]
            self.assertEqual(len(d), initlen - 6)

    def test_indexing_assign_example(self):
        def almost_equal_list(s, t):
            for e, f in zip(s, t):
                self.assertAlmostEqual(e, f)

        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d = data.Table("test2")

            vara = d.domain["a"]
            metaa = d.domain.index(vara)

            self.assertFalse(isnan(d[0, "a"]))
            d[0] = ["3.14", "1", "f"]
            almost_equal_list(d[0].values(), [3.14, "1", "f"])
            self.assertTrue(isnan(d[0, "a"]))
            d[0] = [3.15, 1, "t"]
            almost_equal_list(d[0].values(), [3.15, "0", "t"])
            d[np.int_(0)] = [3.15, 2, "f"]
            almost_equal_list(d[0].values(), [3.15, 2, "f"])

            with self.assertRaises(ValueError):
                d[0] = ["3.14", "1"]

            with self.assertRaises(ValueError):
                d[np.int_(0)] = ["3.14", "1"]

            ex = data.Instance(d.domain, ["3.16", "1", "f"])
            d[0] = ex
            almost_equal_list(d[0].values(), [3.16, "1", "f"])

            ex = data.Instance(d.domain, ["3.16", 2, "t"])
            d[np.int_(0)] = ex
            almost_equal_list(d[0].values(), [3.16, 2, "t"])

            ex = data.Instance(d.domain, ["3.16", "1", "f"])
            ex["e"] = "mmmapp"
            d[0] = ex
            almost_equal_list(d[0].values(), [3.16, "1", "f"])
            self.assertEqual(d[0, "e"], "mmmapp")

    def test_slice(self):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d = data.Table("test2")
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
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d = data.Table("test2")
            d[2:5, 0] = 42
            self.assertEqual([e[0] for e in d],
                             [0, 1.1, 42, 42, 42, 2.25, 2.26, 3.333, Unknown])
            d[:3, "b"] = 43
            self.assertEqual([e[0] for e in d],
                             [43, 43, 43, 42, 42, 2.25, 2.26, 3.333, None])
            d[-2:, d.domain[0]] = 44
            self.assertEqual([e[0] for e in d],
                             [43, 43, 43, 42, 42, 2.25, 2.26, 44, 44])

            d[2:5, "a"] = "A"
            self.assertEqual([e["a"] for e in d], list("ABAAACCDE"))

    def test_del_slice_example(self):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d = data.Table("test2")

            vals = [e[0] for e in d]

            del d[2:2]
            self.assertEqual([e[0] for e in d], vals)

            del d[2:5]
            del vals[2:5]
            self.assertEqual([e[0] for e in d], vals)

            del d[:]
            self.assertEqual(len(d), 0)

    def test_set_slice_example(self):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d = data.Table("test2")
            d[5, 0] = 42
            d[:3] = d[5]
            self.assertEqual(d[1, 0], 42)

            d[5:2:-1] = [3, None, None]
            self.assertEqual([e[0] for e in d],
                             [42, 42, 42, 3, 3, 3, 2.26, 3.333, None])
            self.assertTrue(isnan(d[3, 2]))

            d[2:5] = 42
            self.assertTrue(np.all(d.X[2:5] == 42))
            self.assertEqual(d.Y[2], 0)


    def test_multiple_indices(self):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d = data.Table("test2")

            with self.assertRaises(IndexError):
                x = d[2, 5, 1]

            with self.assertRaises(IndexError):
                x = d[(2, 5, 1)]

            x = d[[2, 5, 1]]
            self.assertEqual([e[0] for e in x], [2.22, 2.25, 1.1])

    def test_assign_multiple_indices_value(self):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d = data.Table("test2")

            d[1:4, "b"] = 42
            self.assertEqual([e[0] for e in d],
                             [0, 42, 42, 42, 2.24, 2.25, 2.26, 3.333, None])

            d[range(5, 2, -1), "b"] = None
            self.assertEqual([e[d.domain[0]] for e in d],
                             [0, 42, 42, None, "?", "", 2.26, 3.333, None])

    def test_del_multiple_indices_example(self):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d = data.Table("test2")

            vals = [e[0] for e in d]

            del d[[1, 5, 2]]
            del vals[5]
            del vals[2]
            del vals[1]
            self.assertEqual([e[0] for e in d], vals)

            del d[range(1, 3)]
            del vals[1:3]
            self.assertEqual([e[0] for e in d], vals)

    def test_set_multiple_indices_example(self):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d = data.Table("test2")

            vals = [e[0] for e in d]
            d[[1, 2, 5]] = [42, None, None]
            vals[1] = vals[2] = vals[5] = 42
            self.assertEqual([e[0] for e in d], vals)

    def test_views(self):
        d = data.Table("zoo")
        crc = d.checksum(True)
        x = d[:20]
        self.assertEqual(crc, d.checksum(True))
        del x[13]
        self.assertEqual(crc, d.checksum(True))
        del x[4:9]
        self.assertEqual(crc, d.checksum(True))

    def test_bool(self):
        d = data.Table("iris")
        self.assertTrue(d)
        del d[:]
        self.assertFalse(d)

        d = data.Table("test3")
        self.assertFalse(d)

        d = data.Table("iris")
        self.assertTrue(d)
        d.clear()
        self.assertFalse(d)

    def test_checksum(self):
        d = data.Table("zoo")
        d[42, 3] = 0
        crc1 = d.checksum(False)
        d[42, 3] = 1
        crc2 = d.checksum(False)
        self.assertNotEqual(crc1, crc2)
        d[42, 3] = 0
        crc3 = d.checksum(False)
        self.assertEqual(crc1, crc3)
        _ = d[42, "name"]
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

        d.set_weights(0)
        d[0].weight = 0.1
        d[10].weight = 0.2
        d[-1].weight = 0.3
        self.assertAlmostEqual(d.total_weight(), 0.6)
        del d[10]
        self.assertAlmostEqual(d.total_weight(), 0.4)
        d.clear()
        self.assertAlmostEqual(d.total_weight(), 0)

    def test_has_missing(self):
        d = data.Table("zoo")
        self.assertFalse(d.has_missing())
        self.assertFalse(d.has_missing_class())

        d[10, 3] = "?"
        self.assertTrue(d.has_missing())
        self.assertFalse(d.has_missing_class())

        d[10].set_class("?")
        self.assertTrue(d.has_missing())
        self.assertTrue(d.has_missing_class())

        d = data.Table("test3")
        self.assertFalse(d.has_missing())
        self.assertFalse(d.has_missing_class())

    def test_shuffle(self):
        d = data.Table("zoo")
        crc = d.checksum()
        names = set(str(x["name"]) for x in d)

        d.shuffle()
        self.assertNotEqual(crc, d.checksum())
        self.assertSetEqual(names, set(str(x["name"]) for x in d))
        crc2 = d.checksum()

        x = d[2:10]
        crcx = x.checksum()
        d.shuffle()
        self.assertNotEqual(crc2, d.checksum())
        self.assertEqual(crcx, x.checksum())

        crc2 = d.checksum()
        x.shuffle()
        self.assertNotEqual(crcx, x.checksum())
        self.assertEqual(crc2, d.checksum())

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

    def test_append(self):
        d = data.Table("test3")
        d.append([None] * 3)
        self.assertEqual(1, len(d))
        self.assertTrue(all(isnan(i) for i in d[0]))

        d.append([42, "0", None])
        self.assertEqual(2, len(d))
        self.assertEqual(d[1], [42, "0", None])

        assert_array_nanequal(
            d.metas,
            np.array([[v.Unknown for v in d.domain.metas]] * 2,
                     dtype=object))

    def test_append2(self):
        d = data.Table("iris")
        d.shuffle()
        l1 = len(d)
        d.append([1, 2, 3, 4, 0])
        self.assertEqual(len(d), l1 + 1)
        self.assertEqual(d[-1], [1, 2, 3, 4, 0])

        x = data.Instance(d[10])
        d.append(x)
        self.assertEqual(d[-1], d[10])

        x = d[:50]
        x.ensure_copy()
        x.append(d[50])
        self.assertEqual(x[50], d[50])

    def test_extend(self):
        d = data.Table("iris")
        d.shuffle()

        x = d[:5]
        x.ensure_copy()
        d.extend(x)
        for i in range(5):
            self.assertEqual(d[i], d[-5 + i])

        y = d[:2, 1]
        x.ensure_copy()
        x.extend(y)
        np.testing.assert_almost_equal(x[-2:, 1].X, y.X)
        self.assertEqual(np.isnan(x).sum(), 8)

    def test_extend2(self):
        d = data.Table("test3")
        d.extend([[None] * 3,
                  [None] * 3])
        assert_array_nanequal(
            d.metas,
            np.array([[v.Unknown for v in d.domain.metas]] * 2,
                     dtype=object))

    def test_copy(self):
        t = data.Table(np.zeros((5, 3)), np.arange(5), np.zeros((5, 3)))

        copy = t.copy()
        self.assertTrue(np.all(t.X == copy.X))
        self.assertTrue(np.all(t.Y == copy.Y))
        self.assertTrue(np.all(t.metas == copy.metas))
        copy[0] = [1, 1, 1, 1, 1, 1, 1, 1]
        self.assertFalse(np.all(t.X == copy.X))
        self.assertFalse(np.all(t.Y == copy.Y))
        self.assertFalse(np.all(t.metas == copy.metas))

    def test_copy_sparse(self):
        t = data.Table('iris')
        t.X = sp.csr_matrix(t.X)
        copy = t.copy()

        self.assertEqual((t.X != copy.X).nnz, 0)      # sparse matrices match by content
        np.testing.assert_equal(t.Y, copy.Y)
        np.testing.assert_equal(t.metas, copy.metas)

        self.assertNotEqual(id(t.X), id(copy.X))
        self.assertNotEqual(id(t._Y), id(copy._Y))
        self.assertNotEqual(id(t.metas), id(copy.metas))

    def test_concatenate(self):
        d1 = data.Domain([data.ContinuousVariable('a1')])
        t1 = data.Table.from_numpy(d1, [[1],
                                        [2]])
        d2 = data.Domain([data.ContinuousVariable('a2')],
                         metas=[data.StringVariable('s')])
        t2 = data.Table.from_numpy(d2, [[3],
                                        [4]], metas=[['foo'],
                                                     ['fuu']])
        self.assertRaises(ValueError,
                          lambda: data.Table.concatenate((t1, t2), axis=5))

        t3 = data.Table.concatenate((t1, t2))
        self.assertEqual(t3.domain.attributes,
                         t1.domain.attributes + t2.domain.attributes)
        self.assertEqual(len(t3.domain.metas), 1)
        self.assertEqual(t3.X.shape, (2, 2))
        self.assertRaises(ValueError, lambda: data.Table.concatenate((t3, t1)))

        t4 = data.Table.concatenate((t3, t3), axis=0)
        np.testing.assert_equal(t4.X, [[1, 3],
                                       [2, 4],
                                       [1, 3],
                                       [2, 4]])
        t4 = data.Table.concatenate((t3, t1), axis=0)
        np.testing.assert_equal(t4.X, [[1, 3],
                                       [2, 4],
                                       [1, np.nan],
                                       [2, np.nan]])

    def test_sparse_concatenate_rows(self):
        iris = Table("iris")
        iris.X = sp.csc_matrix(iris.X)
        new = Table.concatenate([iris, iris], axis=0)
        self.assertEqual(len(new), 300)
        self.assertTrue(sp.issparse(new.X), "Concatenated X is not sparse.")
        self.assertFalse(sp.issparse(new.Y), "Concatenated Y is not dense.")
        self.assertFalse(sp.issparse(new.metas), "Concatenated metas is not dense.")
        self.assertEqual(len(new.ids), 300)

    def test_convert_through_append(self):
        d = data.Table("iris")
        dom2 = data.Domain([d.domain[0], d.domain[2], d.domain[4]])
        d2 = data.Table(dom2)
        dom3 = data.Domain([d.domain[1], d.domain[2]], None)
        d3 = data.Table(dom3)
        for e in d[:5]:
            d2.append(e)
            d3.append(e)
        for e, e2, e3 in zip(d, d2, d3):
            self.assertEqual(e[0], e2[0])
            self.assertEqual(e[1], e3[0])

    def test_pickle(self):
        import pickle

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

    def test_translate_through_slice(self):
        d = data.Table("iris")
        dom = data.Domain(["petal length", "sepal length", "iris"],
                          source=d.domain)
        d_ref = d[:10, dom]
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
        d = data.Table(dom)
        d += [[i] for i in range(3)]
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
        d = data.Table(dom)
        d += [[i] for i in range(3)]
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

        d = data.Table("zoo")
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

    def test_save_pickle(self):
        table = data.Table("iris")
        try:
            table.save("iris.pickle")
            table2 = data.Table.from_file("iris.pickle")
            np.testing.assert_almost_equal(table.X, table2.X)
            np.testing.assert_almost_equal(table.Y, table2.Y)
            self.assertIs(table.domain[0], table2.domain[0])
        finally:
            os.remove("iris.pickle")

    def test_from_numpy(self):
        import random

        a = np.arange(20, dtype="d").reshape((4, 5))
        a[:, -1] = [0, 0, 0, 1]
        dom = data.Domain([data.ContinuousVariable(x) for x in "abcd"],
                          data.DiscreteVariable("e", values=["no", "yes"]))
        table = data.Table(dom, a)
        for i in range(4):
            self.assertEqual(table[i].get_class(), "no" if i < 3 else "yes")
            for j in range(5):
                self.assertEqual(a[i, j], table[i, j])
                table[i, j] = random.random()
                self.assertEqual(a[i, j], table[i, j])

        with self.assertRaises(IndexError):
            table[0, -5] = 5

    def test_filter_is_defined(self):
        d = data.Table("iris")
        d[1, 4] = Unknown
        self.assertTrue(isnan(d[1, 4]))
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
        d[1, 4] = Unknown
        self.assertTrue(isnan(d[1, 4]))
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
        for i in range(5):
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
        data = Table(Domain([], metas=[var]), [[x] for x in range(21)])
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

        f = filter.FilterDiscrete(d.domain.class_var, values=[2, data.Table])
        self.assertRaises(TypeError, d._filter_values, f)

        v = d.columns
        f = filter.FilterDiscrete(v.hair, values=None)
        self.assertEqual(len(filter.Values([f])(d)), len(d))

        d[:5, v.hair] = Unknown
        self.assertEqual(len(filter.Values([f])(d)), len(d) - 5)

    def test_valueFilter_string_is_defined(self):
        d = data.Table("test9.tab")
        f = filter.FilterString(-5, filter.FilterString.IsDefined)
        x = filter.Values([f])(d)
        self.assertEqual(len(x), 7)

    def test_valueFilter_discrete_meta_is_defined(self):
        d = data.Table("test9.tab")
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
        domain_metas = data.Domain(table.domain.attributes,
                                   table.domain.class_vars,
                                   attributes_metas)
        table_metas = data.Table(domain_metas, table.X, table.Y, metas)
        new_table = data.Table(data.Domain(table_metas.domain.metas,
                                           table_metas.domain.metas,
                                           table_metas.domain.metas),
                               table_metas)
        self.assertEqual(new_table.X.dtype, np.float64)
        self.assertEqual(new_table.Y.dtype, np.float64)
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

        table.X = sp.csr_matrix(table.X)
        self.assertTrue(table.is_sparse())

    def test_repr_sparse_with_one_row(self):
        table = data.Table("iris")[:1]
        table.X = sp.csr_matrix(table.X)
        repr(table)     # make sure repr does not crash


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

    data = np.random.random((nrows, len(attributes)))
    class_data = np.random.random((nrows, len(class_vars)))
    meta_data = np.random.random((nrows, len(metas)))
    weight_data = np.random.random((nrows, 1))

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
        class_vars = self.class_vars if with_classes else []
        metas = self.metas if with_metas else []
        variables = attributes + class_vars
        return MagicMock(data.Domain,
                         attributes=attributes,
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
        data.Table(filename=self.filename)

        read_data.assert_called_with(self.filename)


class CreateTableWithUrl(TableTests):
    def test_load_from_url(self):
        d1 = data.Table('iris')
        d2 = data.Table('https://raw.githubusercontent.com/biolab/orange3/master/Orange/datasets/iris.tab')
        np.testing.assert_array_equal(d1.X, d2.X)
        np.testing.assert_array_equal(d1.Y, d2.Y)

    class _MockUrlOpen(MagicMock):
        headers = {'content-disposition': 'attachment; filename="Something-FormResponses.tsv"; '
                                          'filename*=UTF-8''Something%20%28Responses%29.tsv'}
        def __enter__(self): return self

        def __exit__(self, *args, **kwargs): pass

        def read(self): return b'''\
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
        data.Table(domain)

        new_from_domain.assert_called_with(domain)


class CreateTableWithData(TableTests):
    def test_creates_a_table_with_given_X(self):
        # from numpy
        table = data.Table(np.array(self.data))
        self.assertIsInstance(table.domain, data.Domain)
        np.testing.assert_almost_equal(table.X, self.data)

        # from list
        table = data.Table(list(self.data))
        self.assertIsInstance(table.domain, data.Domain)
        np.testing.assert_almost_equal(table.X, self.data)

        # from tuple
        table = data.Table(tuple(self.data))
        self.assertIsInstance(table.domain, data.Domain)
        np.testing.assert_almost_equal(table.X, self.data)

    def test_creates_a_table_from_domain_and_list(self):
        domain = data.Domain([data.DiscreteVariable(name="a", values="mf"),
                              data.ContinuousVariable(name="b")],
                             data.DiscreteVariable(name="y", values="abc"))
        table = data.Table(domain, [[0, 1, 2],
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
        table = data.Table(domain, [[0, 1, 2],
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
        table = data.Table(domain, [[0, 1, 2, "X", 2, "bb"],
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
        new_table = data.Table(table.domain, [d for d in table])
        self.assertIs(table.domain, new_table.domain)
        np.testing.assert_almost_equal(table.X, new_table.X)
        np.testing.assert_almost_equal(table.Y, new_table.Y)
        np.testing.assert_almost_equal(table.W, new_table.W)
        self.assertEqual(table.domain, new_table.domain)
        np.testing.assert_array_equal(table.metas, new_table.metas)

    def test_creates_a_table_from_list_of_instances_with_metas(self):
        table = data.Table('zoo')
        new_table = data.Table(table.domain, [d for d in table])
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
        table = data.Table(self.data, self.class_data)

        self.assertIsInstance(table.domain, data.Domain)
        np.testing.assert_almost_equal(table.X, self.data)
        np.testing.assert_almost_equal(table.Y, self.class_data)

    def test_creates_a_table_with_given_X_Y_and_metas(self):
        table = data.Table(self.data, self.class_data, self.meta_data)

        self.assertIsInstance(table.domain, data.Domain)
        np.testing.assert_almost_equal(table.X, self.data)
        np.testing.assert_almost_equal(table.Y, self.class_data)
        np.testing.assert_almost_equal(table.metas, self.meta_data)

    def test_creates_a_discrete_class_if_Y_has_few_distinct_values(self):
        Y = np.array([float(np.random.randint(0, 2)) for i in self.data])
        table = data.Table(self.data, Y, self.meta_data)

        np.testing.assert_almost_equal(table.Y, Y)
        self.assertIsInstance(table.domain.class_vars[0],
                              data.DiscreteVariable)
        self.assertEqual(table.domain.class_vars[0].values, ["v1", "v2"])

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
        def assert_equal(T1, T2):
            np.testing.assert_array_equal(T1.X, T2.X)
            np.testing.assert_array_equal(T1.Y, T2.Y)
            np.testing.assert_array_equal(T1.metas, T2.metas)
            np.testing.assert_array_equal(T1.W, T2.W)

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
    ]

    row_indices = [1, 5, 6, 7]

    def setUp(self):
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

    def test_can_copy_table(self):
        new_table = data.Table.from_table(self.domain, self.table)
        self.assert_table_with_filter_matches(new_table, self.table)

    def test_can_filter_rows_with_list(self):
        for indices in ([0], [1, 5, 6, 7]):
            new_table = data.Table.from_table(
                self.domain, self.table, row_indices=indices)
            self.assert_table_with_filter_matches(
                new_table, self.table, rows=indices)

    def test_can_filter_row_with_slice(self):
        for slice_ in self.interesting_slices:
            new_table = data.Table.from_table(
                self.domain, self.table, row_indices=slice_)
            self.assert_table_with_filter_matches(
                new_table, self.table, rows=slice_)

    def test_can_use_attributes_as_new_columns(self):
        a, c, m = column_sizes(self.table)
        order = [random.randrange(a) for _ in self.domain.attributes]
        new_attributes = [self.domain.attributes[i] for i in order]
        new_domain = self.create_domain(
            new_attributes, new_attributes, new_attributes)
        new_table = data.Table.from_table(new_domain, self.table)

        self.assert_table_with_filter_matches(
            new_table, self.table, xcols=order, ycols=order, mcols=order)

    def test_can_use_class_vars_as_new_columns(self):
        a, c, m = column_sizes(self.table)
        order = [random.randrange(a, a + c) for _ in self.domain.class_vars]
        new_classes = [self.domain.class_vars[i - a] for i in order]
        new_domain = self.create_domain(new_classes, new_classes, new_classes)
        new_table = data.Table.from_table(new_domain, self.table)

        self.assert_table_with_filter_matches(
            new_table, self.table, xcols=order, ycols=order, mcols=order)

    def test_can_use_metas_as_new_columns(self):
        a, c, m = column_sizes(self.table)
        order = [random.randrange(-m + 1, 0) for _ in self.domain.metas]
        new_metas = [self.domain.metas[::-1][i] for i in order]
        new_domain = self.create_domain(new_metas, new_metas, new_metas)
        new_table = data.Table.from_table(new_domain, self.table)

        self.assert_table_with_filter_matches(
            new_table, self.table, xcols=order, ycols=order, mcols=order)

    def test_can_use_combination_of_all_as_new_columns(self):
        a, c, m = column_sizes(self.table)
        order = ([random.randrange(a) for _ in self.domain.attributes] +
                 [random.randrange(a, a + c) for _ in self.domain.class_vars] +
                 [random.randrange(-m + 1, 0) for _ in self.domain.metas])
        random.shuffle(order)
        vars = list(self.domain.variables) + list(self.domain.metas[::-1])
        vars = [vars[i] for i in order]

        new_domain = self.create_domain(vars, vars, vars)
        new_table = data.Table.from_table(new_domain, self.table)
        self.assert_table_with_filter_matches(
            new_table, self.table, xcols=order, ycols=order, mcols=order)

    def test_creates_table_with_given_domain_and_row_filter(self):
        a, c, m = column_sizes(self.table)
        order = ([random.randrange(a) for _ in self.domain.attributes] +
                 [random.randrange(a, a + c) for _ in self.domain.class_vars] +
                 [random.randrange(-m + 1, 0) for _ in self.domain.metas])
        random.shuffle(order)
        vars = list(self.domain.variables) + list(self.domain.metas[::-1])
        vars = [vars[i] for i in order]

        new_domain = self.create_domain(vars, vars, vars)
        new_table = data.Table.from_table(new_domain, self.table, [0])
        self.assert_table_with_filter_matches(
            new_table, self.table[:1], xcols=order, ycols=order, mcols=order)

        new_table = data.Table.from_table(new_domain, self.table, [2, 1, 0])
        self.assert_table_with_filter_matches(
            new_table, self.table[2::-1], xcols=order, ycols=order, mcols=order)

        new_table = data.Table.from_table(new_domain, self.table, [])
        self.assert_table_with_filter_matches(
            new_table, self.table[:0], xcols=order, ycols=order, mcols=order)

    def test_from_table_sparse_move_some_to_empty_metas(self):
        iris = data.Table("iris")
        iris.X = sp.csr_matrix(iris.X)
        new_domain = data.domain.Domain(
            iris.domain.attributes[:2], iris.domain.class_vars,
            iris.domain.attributes[2:], source=iris.domain)
        new_iris = data.Table.from_table(new_domain, iris)

        self.assertTrue(sp.issparse(new_iris.X))
        self.assertTrue(sp.issparse(new_iris.metas))
        self.assertEqual(new_iris.X.shape, (len(iris), 2))
        self.assertEqual(new_iris.metas.shape, (len(iris), 2))

        # move back
        back_iris = data.Table.from_table(iris.domain, new_iris)
        self.assertEqual(back_iris.domain, iris.domain)
        self.assertTrue(sp.issparse(back_iris.X))
        self.assertTrue(sp.issparse(back_iris.metas))
        self.assertEqual(back_iris.X.shape, iris.X.shape)
        self.assertEqual(back_iris.metas.shape, iris.metas.shape)

    def test_from_table_sparse_move_all_to_empty_metas(self):
        iris = data.Table("iris")
        iris.X = sp.csr_matrix(iris.X)
        new_domain = data.domain.Domain(
            [], iris.domain.class_vars, iris.domain.attributes,
            source=iris.domain)
        new_iris = data.Table.from_table(new_domain, iris)

        self.assertTrue(sp.issparse(new_iris.X))
        self.assertTrue(sp.issparse(new_iris.metas))
        self.assertEqual(new_iris.X.shape, (len(iris), 0))
        self.assertEqual(new_iris.metas.shape, (len(iris), 4))

        # move back
        back_iris = data.Table.from_table(iris.domain, new_iris)
        self.assertEqual(back_iris.domain, iris.domain)
        self.assertTrue(sp.issparse(back_iris.X))
        self.assertTrue(sp.issparse(back_iris.metas))
        self.assertEqual(back_iris.X.shape, iris.X.shape)
        self.assertEqual(back_iris.metas.shape, iris.metas.shape)

    def test_from_table_sparse_move_to_nonempty_metas(self):
        brown = data.Table("brown-selected")
        brown.X = sp.csr_matrix(brown.X)
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
        if Y.shape[1] == 1:
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


class TableElementAssignmentTest(TableTests):
    def setUp(self):
        super().setUp()
        self.domain = \
            self.create_domain(self.attributes, self.class_vars, self.metas)
        self.table = \
            data.Table(self.domain, self.data, self.class_data, self.meta_data)

    def test_can_assign_values(self):
        self.table[0, 0] = 42.
        self.assertAlmostEqual(self.table.X[0, 0], 42.)

    def test_can_assign_values_to_classes(self):
        a, c, m = column_sizes(self.table)
        self.table[0, a] = 42.
        self.assertAlmostEqual(self.table.Y[0], 42.)

    def test_can_assign_values_to_metas(self):
        self.table[0, -1] = 42.
        self.assertAlmostEqual(self.table.metas[0, 0], 42.)

    def test_can_assign_rows_to_rows(self):
        self.table[0] = self.table[1]
        np.testing.assert_almost_equal(
            self.table.X[0], self.table.X[1])
        np.testing.assert_almost_equal(
            self.table.Y[0], self.table.Y[1])
        np.testing.assert_almost_equal(
            self.table.metas[0], self.table.metas[1])

    def test_can_assign_lists(self):
        a, c, m = column_sizes(self.table)
        new_example = [float(i)
                       for i in range(len(self.attributes + self.class_vars))]
        self.table[0] = new_example
        np.testing.assert_almost_equal(
            self.table.X[0], np.array(new_example[:a]))
        np.testing.assert_almost_equal(
            self.table.Y[0], np.array(new_example[a:]))

    def test_can_assign_np_array(self):
        a, c, m = column_sizes(self.table)
        new_example = \
            np.array([float(i)
                      for i in range(len(self.attributes + self.class_vars))])
        self.table[0] = new_example
        np.testing.assert_almost_equal(self.table.X[0], new_example[:a])
        np.testing.assert_almost_equal(self.table.Y[0], new_example[a:])


class InterfaceTest(unittest.TestCase):
    """Basic tests each implementation of Table should pass."""

    features = (
        data.ContinuousVariable(name="Continuous Feature 1"),
        data.ContinuousVariable(name="Continuous Feature 2"),
        data.DiscreteVariable(name="Discrete Feature 1", values=["0", "1"]),
        data.DiscreteVariable(name="Discrete Feature 2",
                              values=["value1", "value2"]),
    )

    class_vars = (
        data.ContinuousVariable(name="Continuous Class"),
        data.DiscreteVariable(name="Discrete Class", values=["m", "f"])
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
        for i in range(self.nrows):
            new_row = [new_value] * len(self.data[i])
            self.table[i] = np.array(new_row)
            self.assertEqual(list(self.table[i]), new_row)

    def test_value_assignment(self):
        new_value = 0.
        for i in range(self.nrows):
            for j in range(len(self.table[i])):
                self.table[i, j] = new_value
                self.assertEqual(self.table[i, j], new_value)

    def test_append_rows(self):
        new_value = 2
        new_row = [new_value] * len(self.data[0])
        self.table.append(new_row)
        self.assertEqual(list(self.table[-1]), new_row)

    def test_insert_rows(self):
        new_value = 2
        new_row = [new_value] * len(self.data[0])
        self.table.insert(0, new_row)
        self.assertEqual(list(self.table[0]), new_row)
        for row, expected in zip(self.table[1:], self.data):
            self.assertEqual(tuple(row), expected)

    def test_insert_view(self):
        new_row = [1] * len(self.data[0])
        tab = self.table[:2]
        self.assertFalse(tab.is_copy())
        tab.insert(0, new_row)
        tab = data.Table.from_numpy(self.table.domain, self.table)
        self.assertFalse(tab.X.flags.c_contiguous)
        tab.insert(0, new_row)

    def test_delete_rows(self):
        for i in range(self.nrows):
            del self.table[0]
            for j in range(len(self.table)):
                self.assertEqual(tuple(self.table[j]), self.data[i + j + 1])

    def test_clear(self):
        self.table.clear()
        self.assertEqual(len(self.table), 0)
        for i in self.table:
            self.fail("Table should not contain any rows.")

    def test_subclasses(self):
        from pathlib import Path

        class _ExtendedTable(data.Table):
            pass

        data_file = _ExtendedTable('iris')
        data_url = _ExtendedTable.from_url(
            Path(os.path.dirname(__file__), 'test1.tab').as_uri())

        self.assertIsInstance(data_file, _ExtendedTable)
        self.assertIsInstance(data_url, _ExtendedTable)


class TestRowInstance(unittest.TestCase):
    def test_assignment(self):
        table = data.Table("zoo")
        inst = table[2]
        self.assertIsInstance(inst, data.RowInstance)

        inst[1] = 0
        self.assertEqual(table[2, 1], 0)
        inst[1] = 1
        self.assertEqual(table[2, 1], 1)

        inst.set_class("mammal")
        self.assertEqual(table[2, len(table.domain.attributes)], "mammal")
        inst.set_class("fish")
        self.assertEqual(table[2, len(table.domain.attributes)], "fish")

        inst[-1] = "Foo"
        self.assertEqual(table[2, -1], "Foo")

    def test_iteration_with_assignment(self):
        table = data.Table("iris")
        for i, row in enumerate(table):
            row[0] = i
        np.testing.assert_array_equal(table.X[:, 0], np.arange(len(table)))

    def test_sparse_assignment(self):
        X = np.eye(4)
        Y = X[2]
        table = data.Table(X, Y)
        row = table[1]
        self.assertFalse(sp.issparse(row.sparse_x))
        self.assertEqual(row[0], 0)
        self.assertEqual(row[1], 1)

        table.X = sp.csr_matrix(table.X)
        table._Y = sp.csr_matrix(table._Y)
        sparse_row = table[1]
        self.assertTrue(sp.issparse(sparse_row.sparse_x))
        self.assertEqual(sparse_row[0], 0)
        self.assertEqual(sparse_row[1], 1)
        sparse_row[1] = 0
        self.assertEqual(sparse_row[1], 0)
        self.assertEqual(table.X[1, 1], 0)
        self.assertEqual(table[2][4], 1)
        table[2][4] = 0
        self.assertEqual(table[2][4], 0)


class TestTableTranspose(unittest.TestCase):
    def test_transpose_no_class(self):
        attrs = [ContinuousVariable("c1"), ContinuousVariable("c2")]
        data = Table(Domain(attrs), np.arange(8).reshape((4, 2)))

        att = [ContinuousVariable("Feature 1"), ContinuousVariable("Feature 2"),
               ContinuousVariable("Feature 3"), ContinuousVariable("Feature 4")]
        domain = Domain(att, metas=[StringVariable("Feature name")])
        result = Table(domain, np.arange(8).reshape((4, 2)).T,
                       metas=np.array(["c1", "c2"])[:, None])

        # transpose and compare
        self._compare_tables(result, Table.transpose(data))

        # transpose of transpose is original
        t = Table.transpose(Table.transpose(data), "Feature name")
        self._compare_tables(data, t)

        # original should not change
        self.assertDictEqual(data.domain.attributes[0].attributes, {})

    def test_transpose_discrete_class(self):
        attrs = [ContinuousVariable("c1"), ContinuousVariable("c2")]
        domain = Domain(attrs, [DiscreteVariable("cls", values=["a", "b"])])
        data = Table(domain, np.arange(8).reshape((4, 2)),
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
        self._compare_tables(result, Table.transpose(data))

        # transpose of transpose is original
        t = Table.transpose(Table.transpose(data), "Feature name")
        self._compare_tables(data, t)

        # original should not change
        self.assertDictEqual(data.domain.attributes[0].attributes, {})

    def test_transpose_continuous_class(self):
        attrs = [ContinuousVariable("c1"), ContinuousVariable("c2")]
        domain = Domain(attrs, [ContinuousVariable("cls")])
        data = Table(domain, np.arange(8).reshape((4, 2)), np.arange(4, 0, -1))

        att = [ContinuousVariable("Feature 1"), ContinuousVariable("Feature 2"),
               ContinuousVariable("Feature 3"), ContinuousVariable("Feature 4")]
        att[0].attributes = {"cls": "4.000"}
        att[1].attributes = {"cls": "3.000"}
        att[2].attributes = {"cls": "2.000"}
        att[3].attributes = {"cls": "1.000"}
        domain = Domain(att, metas=[StringVariable("Feature name")])
        result = Table(domain, np.arange(8).reshape((4, 2)).T,
                       metas=np.array(["c1", "c2"])[:, None])

        # transpose and compare
        self._compare_tables(result, Table.transpose(data))

        # transpose of transpose
        t = Table.transpose(Table.transpose(data), "Feature name")
        self._compare_tables(data, t)

        # original should not change
        self.assertDictEqual(data.domain.attributes[0].attributes, {})

    def test_transpose_missing_class(self):
        attrs = [ContinuousVariable("c1"), ContinuousVariable("c2")]
        domain = Domain(attrs, [ContinuousVariable("cls")])
        data = Table(domain, np.arange(8).reshape((4, 2)),
                     np.array([np.nan, 3, 2, 1]))

        att = [ContinuousVariable("Feature 1"), ContinuousVariable("Feature 2"),
               ContinuousVariable("Feature 3"), ContinuousVariable("Feature 4")]
        att[1].attributes = {"cls": "3.000"}
        att[2].attributes = {"cls": "2.000"}
        att[3].attributes = {"cls": "1.000"}
        domain = Domain(att, metas=[StringVariable("Feature name")])
        result = Table(domain, np.arange(8).reshape((4, 2)).T,
                       metas=np.array(["c1", "c2"])[:, None])

        # transpose and compare
        self._compare_tables(result, Table.transpose(data))

        # transpose of transpose
        t = Table.transpose(Table.transpose(data), "Feature name")
        self._compare_tables(data, t)

        # original should not change
        self.assertDictEqual(data.domain.attributes[0].attributes, {})

    def test_transpose_multiple_class(self):
        attrs = [ContinuousVariable("c1"), ContinuousVariable("c2")]
        class_vars = [ContinuousVariable("cls1"), ContinuousVariable("cls2")]
        domain = Domain(attrs, class_vars)
        data = Table(domain, np.arange(8).reshape((4, 2)),
                     np.arange(8).reshape((4, 2)))

        att = [ContinuousVariable("Feature 1"), ContinuousVariable("Feature 2"),
               ContinuousVariable("Feature 3"), ContinuousVariable("Feature 4")]
        att[0].attributes = {"cls1": "0.000", "cls2": "1.000"}
        att[1].attributes = {"cls1": "2.000", "cls2": "3.000"}
        att[2].attributes = {"cls1": "4.000", "cls2": "5.000"}
        att[3].attributes = {"cls1": "6.000", "cls2": "7.000"}
        domain = Domain(att, metas=[StringVariable("Feature name")])
        result = Table(domain, np.arange(8).reshape((4, 2)).T,
                       metas=np.array(["c1", "c2"])[:, None])

        # transpose and compare
        self._compare_tables(result, Table.transpose(data))

        # transpose of transpose
        t = Table.transpose(Table.transpose(data), "Feature name")
        self._compare_tables(data, t)

        # original should not change
        self.assertDictEqual(data.domain.attributes[0].attributes, {})

    def test_transpose_metas(self):
        attrs = [ContinuousVariable("c1"), ContinuousVariable("c2")]
        metas = [StringVariable("m1")]
        domain = Domain(attrs, metas=metas)
        X = np.arange(8).reshape((4, 2))
        M = np.array(["aa", "bb", "cc", "dd"])[:, None]
        data = Table(domain, X, metas=M)

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
        self._compare_tables(result, Table.transpose(data))

        # transpose of transpose
        t = Table.transpose(Table.transpose(data), "Feature name")
        self._compare_tables(data, t)

        # original should not change
        self.assertDictEqual(data.domain.attributes[0].attributes, {})

    def test_transpose_discrete_metas(self):
        attrs = [ContinuousVariable("c1"), ContinuousVariable("c2")]
        metas = [DiscreteVariable("m1", values=["aa", "bb"])]
        domain = Domain(attrs, metas=metas)
        X = np.arange(8).reshape((4, 2))
        M = np.array([0, 1, 0, 1])[:, None]
        data = Table(domain, X, metas=M)

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
        self._compare_tables(result, Table.transpose(data))

        # transpose of transpose is original
        t = Table.transpose(Table.transpose(data), "Feature name")
        self._compare_tables(data, t)

        # original should not change
        self.assertDictEqual(data.domain.attributes[0].attributes, {})

    def test_transpose_continuous_metas(self):
        attrs = [ContinuousVariable("c1"), ContinuousVariable("c2")]
        metas = [ContinuousVariable("m1")]
        domain = Domain(attrs, metas=metas)
        X = np.arange(8).reshape((4, 2))
        M = np.array([0.0, 1.0, 0.0, 1.0])[:, None]
        data = Table(domain, X, metas=M)

        att = [ContinuousVariable("Feature 1"), ContinuousVariable("Feature 2"),
               ContinuousVariable("Feature 3"), ContinuousVariable("Feature 4")]
        att[0].attributes = {"m1": "0.000"}
        att[1].attributes = {"m1": "1.000"}
        att[2].attributes = {"m1": "0.000"}
        att[3].attributes = {"m1": "1.000"}
        domain = Domain(att, metas=[StringVariable("Feature name")])
        result = Table(domain, np.arange(8).reshape((4, 2)).T,
                       metas=np.array(["c1", "c2"])[:, None])

        # transpose and compare
        self._compare_tables(result, Table.transpose(data))

        # transpose of transpose is original
        t = Table.transpose(Table.transpose(data), "Feature name")
        self._compare_tables(data, t)

        # original should not change
        self.assertDictEqual(data.domain.attributes[0].attributes, {})

    def test_transpose_missing_metas(self):
        attrs = [ContinuousVariable("c1"), ContinuousVariable("c2")]
        metas = [StringVariable("m1")]
        domain = Domain(attrs, metas=metas)
        X = np.arange(8).reshape((4, 2))
        M = np.array(["aa", "bb", "", "dd"])[:, None]
        data = Table(domain, X, metas=M)

        att = [ContinuousVariable("Feature 1"), ContinuousVariable("Feature 2"),
               ContinuousVariable("Feature 3"), ContinuousVariable("Feature 4")]
        att[0].attributes = {"m1": "aa"}
        att[1].attributes = {"m1": "bb"}
        att[3].attributes = {"m1": "dd"}
        domain = Domain(att, metas=[StringVariable("Feature name")])
        result = Table(domain, np.arange(8).reshape((4, 2)).T,
                       metas=np.array(["c1", "c2"])[:, None])

        # transpose and compare
        self._compare_tables(result, Table.transpose(data))

        # transpose of transpose
        t = Table.transpose(Table.transpose(data), "Feature name")
        self._compare_tables(data, t)

        # original should not change
        self.assertDictEqual(data.domain.attributes[0].attributes, {})

    def test_transpose_multiple_metas(self):
        attrs = [ContinuousVariable("c1"), ContinuousVariable("c2")]
        metas = [StringVariable("m1"), StringVariable("m2")]
        domain = Domain(attrs, metas=metas)
        X = np.arange(8).reshape((4, 2))
        M = np.array([["aa", "aaa"], ["bb", "bbb"],
                      ["cc", "ccc"], ["dd", "ddd"]])
        data = Table(domain, X, metas=M)

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
        self._compare_tables(result, Table.transpose(data))

        # transpose of transpose
        t = Table.transpose(Table.transpose(data), "Feature name")
        self._compare_tables(data, t)

        # original should not change
        self.assertDictEqual(data.domain.attributes[0].attributes, {})

    def test_transpose_class_and_metas(self):
        attrs = [ContinuousVariable("c1"), ContinuousVariable("c2")]
        metas = [StringVariable("m1"), StringVariable("m2")]
        domain = Domain(attrs, [ContinuousVariable("cls")], metas)
        M = np.array([["aa", "aaa"], ["bb", "bbb"],
                      ["cc", "ccc"], ["dd", "ddd"]])
        data = Table(domain, np.arange(8).reshape((4, 2)), np.arange(1, 5), M)

        att = [ContinuousVariable("Feature 1"), ContinuousVariable("Feature 2"),
               ContinuousVariable("Feature 3"), ContinuousVariable("Feature 4")]
        att[0].attributes = {"cls": "1.000", "m1": "aa", "m2": "aaa"}
        att[1].attributes = {"cls": "2.000", "m1": "bb", "m2": "bbb"}
        att[2].attributes = {"cls": "3.000", "m1": "cc", "m2": "ccc"}
        att[3].attributes = {"cls": "4.000", "m1": "dd", "m2": "ddd"}
        domain = Domain(att, metas=[StringVariable("Feature name")])
        result = Table(domain, np.arange(8).reshape((4, 2)).T,
                       metas=np.array(["c1", "c2"])[:, None])

        # transpose and compare
        self._compare_tables(result, Table.transpose(data))

        # transpose of transpose
        t = Table.transpose(Table.transpose(data), "Feature name")
        self._compare_tables(data, t)

        # original should not change
        self.assertDictEqual(data.domain.attributes[0].attributes, {})

    def test_transpose_attributes_of_attributes_discrete(self):
        attrs = [ContinuousVariable("c1"), ContinuousVariable("c2")]
        attrs[0].attributes = {"attr1": "a", "attr2": "aa"}
        attrs[1].attributes = {"attr1": "b", "attr2": "bb"}
        domain = Domain(attrs)
        data = Table(domain, np.arange(8).reshape((4, 2)))

        att = [ContinuousVariable("Feature 1"), ContinuousVariable("Feature 2"),
               ContinuousVariable("Feature 3"), ContinuousVariable("Feature 4")]
        metas = [StringVariable("Feature name"),
                 DiscreteVariable("attr1", values=("a", "b")),
                 DiscreteVariable("attr2", values=("aa", "bb"))]
        domain = Domain(att, metas=metas)
        M = np.array([["c1", 0.0, 0.0], ["c2", 1.0, 1.0]], dtype=object)
        result = Table(domain, np.arange(8).reshape((4, 2)).T, metas=M)

        # transpose and compare
        self._compare_tables(result, Table.transpose(data))

        # transpose of transpose
        t = Table.transpose(Table.transpose(data), "Feature name")
        self._compare_tables(data, t)

        # original should not change
        self.assertDictEqual(data.domain.attributes[0].attributes,
                             {"attr1": "a", "attr2": "aa"})

    def test_transpose_attributes_of_attributes_continuous(self):
        attrs = [ContinuousVariable("c1"), ContinuousVariable("c2")]
        attrs[0].attributes = {"attr1": "1.100", "attr2": "1.300"}
        attrs[1].attributes = {"attr1": "2.200", "attr2": "2.300"}
        domain = Domain(attrs)
        data = Table(domain, np.arange(8).reshape((4, 2)))

        att = [ContinuousVariable("Feature 1"), ContinuousVariable("Feature 2"),
               ContinuousVariable("Feature 3"), ContinuousVariable("Feature 4")]
        metas = [StringVariable("Feature name"), ContinuousVariable("attr1"),
                 ContinuousVariable("attr2")]
        domain = Domain(att, metas=metas)
        result = Table(domain, np.arange(8).reshape((4, 2)).T,
                       metas=np.array([["c1", 1.1, 1.3],
                                       ["c2", 2.2, 2.3]], dtype=object))

        # transpose and compare
        self._compare_tables(result, Table.transpose(data))

        # transpose of transpose
        t = Table.transpose(Table.transpose(data), "Feature name")
        self._compare_tables(data, t)

        # original should not change
        self.assertDictEqual(data.domain.attributes[0].attributes,
                             {"attr1": "1.100", "attr2": "1.300"})

    def test_transpose_attributes_of_attributes_missings(self):
        attrs = [ContinuousVariable("c1"), ContinuousVariable("c2")]
        attrs[0].attributes = {"attr1": "a", "attr2": "aa"}
        attrs[1].attributes = {"attr1": "b"}
        domain = Domain(attrs)
        data = Table(domain, np.arange(8).reshape((4, 2)))

        att = [ContinuousVariable("Feature 1"), ContinuousVariable("Feature 2"),
               ContinuousVariable("Feature 3"), ContinuousVariable("Feature 4")]
        metas = [StringVariable("Feature name"),
                 DiscreteVariable("attr1", values=("a", "b")),
                 DiscreteVariable("attr2", values=("aa",))]
        domain = Domain(att, metas=metas)
        M = np.array([["c1", 0.0, 0.0], ["c2", 1.0, np.nan]], dtype=object)
        result = Table(domain, np.arange(8).reshape((4, 2)).T, metas=M)

        # transpose and compare
        self._compare_tables(result, Table.transpose(data))

        # transpose of transpose
        t = Table.transpose(Table.transpose(data), "Feature name")
        self._compare_tables(data, t)

        # original should not change
        self.assertDictEqual(data.domain.attributes[0].attributes,
                             {"attr1": "a", "attr2": "aa"})

    def test_transpose_class_metas_attributes(self):
        attrs = [ContinuousVariable("c1"), ContinuousVariable("c2")]
        attrs[0].attributes = {"attr1": "a1", "attr2": "aa1"}
        attrs[1].attributes = {"attr1": "b1", "attr2": "bb1"}
        metas = [StringVariable("m1"), StringVariable("m2")]
        domain = Domain(attrs, [ContinuousVariable("cls")], metas)
        M = np.array([["aa", "aaa"], ["bb", "bbb"],
                      ["cc", "ccc"], ["dd", "ddd"]])
        data = Table(domain, np.arange(8).reshape((4, 2)), np.arange(1, 5), M)

        att = [ContinuousVariable("Feature 1"), ContinuousVariable("Feature 2"),
               ContinuousVariable("Feature 3"), ContinuousVariable("Feature 4")]
        att[0].attributes = {"cls": "1.000", "m1": "aa", "m2": "aaa"}
        att[1].attributes = {"cls": "2.000", "m1": "bb", "m2": "bbb"}
        att[2].attributes = {"cls": "3.000", "m1": "cc", "m2": "ccc"}
        att[3].attributes = {"cls": "4.000", "m1": "dd", "m2": "ddd"}
        metas = [StringVariable("Feature name"),
                 DiscreteVariable("attr1", values=("a1", "b1")),
                 DiscreteVariable("attr2", values=("aa1", "bb1"))]
        domain = Domain(att, metas=metas)
        M = np.array([["c1", 0.0, 0.0], ["c2", 1.0, 1.0]], dtype=object)
        result = Table(domain, np.arange(8).reshape((4, 2)).T, metas=M)

        # transpose and compare
        self._compare_tables(result, Table.transpose(data))

        # transpose of transpose
        t = Table.transpose(Table.transpose(data), "Feature name")
        self._compare_tables(data, t)

        # original should not change
        self.assertDictEqual(data.domain.attributes[0].attributes,
                             {"attr1": "a1", "attr2": "aa1"})

    def test_transpose(self):
        zoo = Table("zoo")
        t1 = Table.transpose(zoo)
        t2 = Table.transpose(t1, "Feature name")
        t3 = Table.transpose(t2)
        self._compare_tables(zoo, t2)
        self._compare_tables(t1, t3)

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


if __name__ == "__main__":
    unittest.main()

