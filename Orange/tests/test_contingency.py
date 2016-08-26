# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring

import unittest
from unittest.mock import Mock

import numpy as np
import scipy.sparse as sp

from Orange.statistics import contingency
from Orange import data
from Orange.tests import test_filename


class TestDiscrete(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.zoo = data.Table("zoo")

    def test_discrete(self):
        cont = contingency.Discrete(self.zoo, 0)
        np.testing.assert_almost_equal(cont["amphibian"], [4, 0])
        np.testing.assert_almost_equal(cont,
                                       [[4, 0], [20, 0], [13, 0],
                                        [4, 4], [10, 0], [2, 39], [5, 0]])

        cont = contingency.Discrete(self.zoo, "predator")
        np.testing.assert_almost_equal(cont["fish"], [4, 9])
        np.testing.assert_almost_equal(cont,
                                       [[1, 3], [11, 9], [4, 9],
                                        [7, 1], [2, 8], [19, 22], [1, 4]])

        cont = contingency.Discrete(self.zoo, self.zoo.domain["predator"])
        np.testing.assert_almost_equal(cont["fish"], [4, 9])
        np.testing.assert_almost_equal(cont,
                                       [[1, 3], [11, 9], [4, 9],
                                        [7, 1], [2, 8], [19, 22], [1, 4]])
        self.assertEqual(cont.unknown_rows, 0)

    def test_discrete_missing(self):
        d = data.Table("zoo")
        d.loc[d.index[25], d.domain.class_vars] = float("nan")
        d.loc[d.index[0], d.domain[0]] = float("nan")
        cont = contingency.Discrete(d, 0)
        np.testing.assert_almost_equal(cont["amphibian"], [3, 0])
        np.testing.assert_almost_equal(cont,
                                       [[3, 0], [20, 0], [13, 0], [4, 4], [10, 0], [2, 38], [5, 0]])
        np.testing.assert_almost_equal(cont.unknowns,
                                       [0, 0, 0, 0, 0, 1, 0])
        self.assertEqual(cont.unknown_rows, 1)

        d = data.Table("zoo")
        d.loc[d.index[2], d.domain.class_vars] = float("nan")
        d.loc[d.index[2], "predator"] = float("nan")
        cont = contingency.Discrete(d, "predator")
        np.testing.assert_almost_equal(cont["fish"], [4, 8])
        np.testing.assert_almost_equal(cont,
                                       [[1, 3], [11, 9], [4, 8], [7, 1], [2, 8], [19, 22], [1, 4]])
        self.assertEqual(cont.unknown_rows, 1)
        np.testing.assert_almost_equal(cont.unknowns, [0, 0, 0, 0, 0, 0, 0])

    def test_discrete_with_fallback(self):
        d = data.Table("zoo")
        d.loc[d.index[25], d.domain.class_vars] = None
        default = contingency.Discrete(d, 0)

        d._compute_contingency = Mock(side_effect=NotImplementedError)
        fallback = contingency.Discrete(d, 0)

        np.testing.assert_almost_equal(fallback, default)
        np.testing.assert_almost_equal(fallback.unknowns, default.unknowns)
        np.testing.assert_almost_equal(fallback.unknown_rows, default.unknown_rows)

    def test_continuous(self):
        d = data.Table("iris")
        cont = contingency.Continuous(d, "sepal width")
        correct = [[2.3, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7,
                    3.8, 3.9, 4.0, 4.1, 4.2, 4.4],
                   [1, 1, 6, 5, 5, 2, 9, 6, 2, 3, 4, 2, 1, 1, 1, 1]]
        np.testing.assert_almost_equal(cont.unknowns, [0, 0, 0])
        np.testing.assert_almost_equal(cont["Iris-setosa"], correct)
        self.assertEqual(cont.unknown_rows, 0)

        correct = [[2.2, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.6, 3.8],
                   [1, 4, 2, 4, 8, 2, 12, 4, 5, 3, 2, 1, 2]]
        np.testing.assert_almost_equal(cont[d.domain.class_var.values.index("Iris-virginica")], correct)
        np.testing.assert_almost_equal(cont.unknowns, [0, 0, 0])
        self.assertEqual(cont.unknown_rows, 0)

    def test_continuous_missing(self):
        d = data.Table("iris")
        d.iloc[1, 1] = float("nan")
        cont = contingency.Continuous(d, "sepal width")
        correct = [[2.3, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7,
                    3.8, 3.9, 4.0, 4.1, 4.2, 4.4],
                   [1, 1, 5, 5, 5, 2, 9, 6, 2, 3, 4, 2, 1, 1, 1, 1]]
        np.testing.assert_almost_equal(cont.unknowns, [1, 0, 0])
        np.testing.assert_almost_equal(cont["Iris-setosa"], correct)
        self.assertEqual(cont.unknown_rows, 0)

        d.loc[d.index[0], d.domain.class_vars] = float("nan")
        cont = contingency.Continuous(d, "sepal width")
        correct = [[2.2, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.6, 3.8],
                   [1, 4, 2, 4, 8, 2, 12, 4, 5, 3, 2, 1, 2]]
        np.testing.assert_almost_equal(cont[d.domain.class_var.values.index("Iris-virginica")], correct)
        np.testing.assert_almost_equal(cont.unknowns, [1, 0, 0])
        self.assertEqual(cont.unknown_rows, 1)

        d.loc[d.index[1], d.domain.class_vars] = float("nan")
        cont = contingency.Continuous(d, "sepal width")
        np.testing.assert_almost_equal(cont.unknowns, [0, 0, 0])
        self.assertEqual(cont.unknown_rows, 2)

    def test_mixedtype_metas(self):
        import Orange
        zoo = Orange.data.Table("zoo")
        dom = Orange.data.Domain(zoo.domain.attributes, zoo.domain.class_var,
                                 zoo.domain.metas + zoo.domain.attributes[:2])
        t = Orange.data.Table(dom, zoo)
        cont = contingency.get_contingency(zoo, 2, t.domain.metas[1])
        np.testing.assert_almost_equal(cont[1], [38, 5])
        np.testing.assert_almost_equal(cont, [[4, 54],
                                              [38, 5]])
        zoo.loc[zoo.index[25], t.domain.metas[1]] = float("nan")
        zoo.loc[zoo.index[0], zoo.domain[2]] = float("nan")
        cont = contingency.get_contingency(zoo, 2, t.domain.metas[1])
        np.testing.assert_almost_equal(cont[1], [37, 5])
        np.testing.assert_almost_equal(cont, [[4, 53],
                                              [37, 5]])
        np.testing.assert_almost_equal(cont.unknowns, [0, 1])
        self.assertEqual(cont.unknown_rows, 1)

    @staticmethod
    def _construct_sparse():
        domain = data.Domain(
            [data.DiscreteVariable("d%i" % i, values=[0, 1, 2])
             for i in range(10)] +
            [data.ContinuousVariable("c%i" % i) for i in range(10)],
            data.DiscreteVariable("y", values=list("abc")))

        #  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19
        #------------------------------------------------------------
        #     2     2  1  1  2        1           1  1     2  0  2
        #        1  1  0  0  1     2                 2     1  0
        #           1     2  0
        #
        #        2        0  1                   1.1
        #
        sdata = np.array([2, 2, 1, 1, 2, 1, 1, 1, 2, 0, 2,
                          1, 1, 0, 0, 1, 2, 2, 1, 0,
                          1, 2, 0,
                          2, 0, 1, 1.1])
        indices = [1, 3, 4, 5, 6, 9, 13, 14, 16, 17, 18,
                   2, 3, 4, 5, 6, 8, 14, 16, 17,
                   3, 5, 6,
                   2, 5, 6, 13]
        indptr = [0, 11, 20, 23, 23, 27]
        X = sp.csr_matrix((sdata, indices, indptr), shape=(5, 20))
        Y = np.array([["b", "c", "b", "a", "a"]]).T
        return data.Table.from_numpy(domain, X, Y)

    def test_sparse(self):
        d = self._construct_sparse()
        cont = contingency.Discrete(d, 5)
        np.testing.assert_almost_equal(cont[0], [1, 0, 0])
        np.testing.assert_almost_equal(cont[1], [0, 1, 1])
        np.testing.assert_almost_equal(cont[2], [1, 0, 0])

        cont = contingency.Continuous(d, 14)
        np.testing.assert_almost_equal(cont[0], [[], []])
        np.testing.assert_almost_equal(cont[1], [[1], [1]])
        np.testing.assert_almost_equal(cont[2], [[2], [1]])

        cont = contingency.Continuous(d, "c3")
        np.testing.assert_almost_equal(cont[0], [[1.1], [1]])
        np.testing.assert_almost_equal(cont[1], [[1], [1]])
        np.testing.assert_almost_equal(cont[2], [[], []])

        cont = contingency.Continuous(d, 12)
        np.testing.assert_almost_equal(cont[0], [[], []])
        np.testing.assert_almost_equal(cont[1], [[], []])
        np.testing.assert_almost_equal(cont[2], [[], []])

    def test_get_contingency(self):
        d = self._construct_sparse()
        cont = contingency.get_contingency(d, 5)
        self.assertIsInstance(cont, contingency.Discrete)
        np.testing.assert_almost_equal(cont[0], [1, 0, 0])
        np.testing.assert_almost_equal(cont[1], [0, 1, 1])
        np.testing.assert_almost_equal(cont[2], [1, 0, 0])

        cont = contingency.get_contingency(d, "c4")
        self.assertIsInstance(cont, contingency.Continuous)
        np.testing.assert_almost_equal(cont[0], [[], []])
        np.testing.assert_almost_equal(cont[1], [[1], [1]])
        np.testing.assert_almost_equal(cont[2], [[2], [1]])

        cont = contingency.get_contingency(d, d.domain[13])
        self.assertIsInstance(cont, contingency.Continuous)
        np.testing.assert_almost_equal(cont[0], [[1.1], [1]])
        np.testing.assert_almost_equal(cont[1], [[1], [1]])
        np.testing.assert_almost_equal(cont[2], [[], []])

    def test_get_contingencies(self):
        d = self._construct_sparse()
        conts = contingency.get_contingencies(d)

        self.assertEqual(len(conts), 20)

        cont = conts[5]
        self.assertIsInstance(cont, contingency.Discrete)
        np.testing.assert_almost_equal(cont[0], [1, 0, 0])
        np.testing.assert_almost_equal(cont[1], [0, 1, 1])
        np.testing.assert_almost_equal(cont[2], [1, 0, 0])

        cont = conts[14]
        self.assertIsInstance(cont, contingency.Continuous)
        np.testing.assert_almost_equal(cont[0], [[], []])
        np.testing.assert_almost_equal(cont[1], [[1], [1]])
        np.testing.assert_almost_equal(cont[2], [[2], [1]])

        conts = contingency.get_contingencies(d, skip_discrete=True)
        self.assertEqual(len(conts), 10)
        cont = conts[4]
        self.assertIsInstance(cont, contingency.Continuous)
        np.testing.assert_almost_equal(cont[0], [[], []])
        np.testing.assert_almost_equal(cont[1], [[1], [1]])
        np.testing.assert_almost_equal(cont[2], [[2], [1]])

        conts = contingency.get_contingencies(d, skip_continuous=True)
        self.assertEqual(len(conts), 10)
        cont = conts[5]
        self.assertIsInstance(cont, contingency.Discrete)
        np.testing.assert_almost_equal(cont[0], [1, 0, 0])
        np.testing.assert_almost_equal(cont[1], [0, 1, 1])
        np.testing.assert_almost_equal(cont[2], [1, 0, 0])

    def test_compute_contingency_metas(self):
        d = data.Table(test_filename("test9.tab"))
        var1, var2 = d.domain[-2], d.domain[-4]
        cont, _ = d._compute_contingency([var1], var2)[0][0]
        np.testing.assert_almost_equal(cont, [[3, 0, 0], [0, 2, 0],
                                              [0, 0, 2], [0, 1, 0]])

    def test_discrete_normalize(self):
        cont1 = contingency.Discrete(self.zoo, 0)
        cont2 = contingency.Discrete(self.zoo, 0)
        cont1.normalize()
        np.testing.assert_array_equal(cont1, cont2 / np.sum(cont2))

    def test_continuous_normalize(self):
        d = data.Table("iris")
        cont = contingency.Continuous(d, "sepal width")
        # assert not throws
        cont.normalize()
        cont.normalize(axis=1)
        with self.assertRaises(ValueError):
            cont.normalize(axis=0)

    def test_setitem_not_implemented(self):
        d = data.Table("iris")
        cont = contingency.Continuous(d, "sepal width")
        with self.assertRaises(NotImplementedError):
            cont[1] = [1, 34, 4, 3, 1]

    def test_equality(self):
        d1 = data.Table("iris")
        cont1 = contingency.Continuous(d1, "sepal width")
        d2 = data.Table("iris")
        cont2 = contingency.Continuous(d2, "sepal width")
        self.assertTrue(cont1 == cont2)
