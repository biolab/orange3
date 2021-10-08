# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring
import copy
import unittest
from unittest.mock import Mock

import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix, csc_matrix

from Orange.data import DiscreteVariable, Table, Domain
from Orange.statistics import contingency
from Orange import data
from Orange.tests import test_filename


def assert_dist_equal(dist, expected):
    np.testing.assert_array_equal(np.asarray(dist), expected)


def assert_dist_almost_equal(dist, expected):
    np.testing.assert_array_almost_equal(np.asarray(dist), expected)


class TestDiscrete(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.zoo = data.Table("zoo")
        cls.test9 = data.Table(test_filename("datasets/test9.tab"))

    def test_discrete(self):
        cont = contingency.Discrete(self.zoo, 0)
        assert_dist_equal(cont["amphibian"], [4, 0])
        assert_dist_equal(cont, [[4, 0], [20, 0], [13, 0],
                                 [4, 4], [10, 0], [2, 39], [5, 0]])

        cont = contingency.Discrete(self.zoo, "predator")
        assert_dist_equal(cont["fish"], [4, 9])
        assert_dist_equal(cont, [[1, 3], [11, 9], [4, 9], [7, 1],
                                 [2, 8], [19, 22], [1, 4]])

        cont = contingency.Discrete(self.zoo, self.zoo.domain["predator"])
        assert_dist_equal(cont["fish"], [4, 9])
        assert_dist_equal(cont, [[1, 3], [11, 9], [4, 9], [7, 1],
                                 [2, 8], [19, 22], [1, 4]])
        self.assertEqual(sum(cont.col_unknowns), 0)
        self.assertEqual(sum(cont.row_unknowns), 0)

    def test_discrete_missing(self):
        d = data.Table("zoo")
        with d.unlocked():
            d.Y[25] = float("nan")
            d[0][0] = float("nan")
        cont = contingency.Discrete(d, 0)
        assert_dist_equal(cont["amphibian"], [3, 0])
        assert_dist_equal(cont, [[3, 0], [20, 0], [13, 0], [4, 4],
                                 [10, 0], [2, 38], [5, 0]])
        np.testing.assert_almost_equal(cont.col_unknowns,
                                       [0, 0, 0, 0, 0, 1, 0])
        np.testing.assert_almost_equal(cont.row_unknowns,
                                       [1, 0])

        d = data.Table("zoo")
        with d.unlocked():
            d.Y[2] = float("nan")
            d[2]["predator"] = float("nan")
        cont = contingency.Discrete(d, "predator")
        assert_dist_equal(cont["fish"], [4, 8])
        assert_dist_equal(cont, [[1, 3], [11, 9], [4, 8], [7, 1],
                                 [2, 8], [19, 22], [1, 4]])
        np.testing.assert_almost_equal(
            cont.col_unknowns, [0, 0, 0, 0, 0, 0, 0])
        np.testing.assert_almost_equal(cont.row_unknowns, [0, 0])
        self.assertEqual(1, cont.unknowns)

    def test_deepcopy(self):
        cont = contingency.Discrete(self.zoo, 0)
        dc = copy.deepcopy(cont)
        self.assertEqual(dc, cont)
        self.assertEqual(dc.col_variable, cont.col_variable)
        self.assertEqual(dc.row_variable, cont.row_variable)

    def test_array_with_unknowns(self):
        d = data.Table("zoo")
        with d.unlocked():
            d.Y[2] = float("nan")
            d.Y[6] = float("nan")
            d[2]["predator"] = float("nan")
            d[4]["predator"] = float("nan")
        cont = contingency.Discrete(d, "predator")
        assert_dist_equal(cont.array_with_unknowns,
                          [[1, 3, 0], [11, 9, 0], [4, 8, 0], [7, 1, 0],
                           [2, 8, 0], [18, 21, 1], [1, 4, 0], [1, 0, 1]])

    def test_discrete_with_fallback(self):
        d = data.Table("zoo")
        with d.unlocked():
            d.Y[25] = None
            d.Y[24] = None
            d.X[0, 0] = None
            d.X[24, 0] = None
        default = contingency.Discrete(d, 0)

        d._compute_contingency = Mock(side_effect=NotImplementedError)
        fallback = contingency.Discrete(d, 0)

        np.testing.assert_array_equal(
            np.asarray(fallback), np.asarray(default))
        np.testing.assert_array_equal(fallback.unknowns, default.unknowns)
        np.testing.assert_array_equal(
            fallback.row_unknowns, default.row_unknowns)
        np.testing.assert_array_equal(
            fallback.col_unknowns, default.col_unknowns)

    def test_continuous(self):
        d = data.Table("iris")
        cont = contingency.Continuous(d, "sepal width")
        correct = [[2.3, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7,
                    3.8, 3.9, 4.0, 4.1, 4.2, 4.4],
                   [1, 1, 6, 5, 5, 2, 9, 6, 2, 3, 4, 2, 1, 1, 1, 1]]

        np.testing.assert_almost_equal(cont.col_unknowns, [0, 0, 0])
        np.testing.assert_almost_equal(cont.row_unknowns, np.zeros(23))
        np.testing.assert_almost_equal(cont["Iris-setosa"], correct)
        self.assertEqual(cont.unknowns, 0)

        correct = [[2.2, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.6, 3.8],
                   [1, 4, 2, 4, 8, 2, 12, 4, 5, 3, 2, 1, 2]]
        np.testing.assert_almost_equal(
            cont[d.domain.class_var.values.index("Iris-virginica")], correct)
        np.testing.assert_almost_equal(cont.col_unknowns, [0, 0, 0])
        np.testing.assert_almost_equal(cont.row_unknowns, np.zeros(23))
        self.assertEqual(cont.unknowns, 0)

    def test_continuous_missing(self):
        d = data.Table("iris")
        with d.unlocked():
            d[1][1] = float("nan")
        cont = contingency.Continuous(d, "sepal width")
        correct = [[2.3, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7,
                    3.8, 3.9, 4.0, 4.1, 4.2, 4.4],
                   [1, 1, 5, 5, 5, 2, 9, 6, 2, 3, 4, 2, 1, 1, 1, 1]]
        np.testing.assert_almost_equal(cont.col_unknowns, [1, 0, 0])
        np.testing.assert_almost_equal(cont.row_unknowns, np.zeros(23))
        np.testing.assert_almost_equal(cont["Iris-setosa"], correct)
        self.assertEqual(cont.unknowns, 0)

        with d.unlocked():
            d.Y[0] = float("nan")
        cont = contingency.Continuous(d, "sepal width")
        correct = [[2.2, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.6, 3.8],
                   [1, 4, 2, 4, 8, 2, 12, 4, 5, 3, 2, 1, 2]]
        np.testing.assert_almost_equal(
            cont[d.domain.class_var.values.index("Iris-virginica")], correct)
        np.testing.assert_almost_equal(cont.col_unknowns, [1, 0, 0])
        np.testing.assert_almost_equal(
            cont.row_unknowns,
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
             0., 0., 0., 0., 0., 0., 0.])
        self.assertEqual(cont.unknowns, 0)

        with d.unlocked():
            d.Y[1] = float("nan")
        cont = contingency.Continuous(d, "sepal width")
        np.testing.assert_almost_equal(cont.col_unknowns, [0, 0, 0])
        np.testing.assert_almost_equal(
            cont.row_unknowns,
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
             0., 0., 0., 0., 0., 0., 0.])
        self.assertEqual(cont.unknowns, 1)

        # this one was failing before since the issue in _contingecy.pyx
        with d.unlocked():
            d.Y[:50] = np.zeros(50) * float("nan")
        cont = contingency.Continuous(d, "sepal width")
        np.testing.assert_almost_equal(cont.col_unknowns, [0, 0, 0])
        np.testing.assert_almost_equal(
            cont.row_unknowns,
            [0., 0., 1., 0., 0., 0., 0., 0., 1., 5., 5., 5., 2., 9., 6., 2.,
             3., 4., 2., 1., 1., 1., 1.])
        self.assertEqual(cont.unknowns, 1)

    @staticmethod
    def test_continuous_array_with_unknowns():
        """
        Test array_with_unknowns function
        """
        d = data.Table("iris")
        with d.unlocked():
            d.Y[:50] = np.zeros(50) * float("nan")
        cont = contingency.Continuous(d, "sepal width")
        correct_row_unknowns = [0., 0., 1., 0., 0., 0., 0., 0., 1., 6., 5., 5.,
                                2., 9., 6., 2., 3., 4., 2., 1., 1., 1., 1.]
        correct_row_unknowns_no_zero = [
            c for c in correct_row_unknowns if c > 0]
        correct_values_no_zero = [
            v for v, c in zip(cont.values, correct_row_unknowns) if c > 0]

        np.testing.assert_almost_equal(cont.row_unknowns, correct_row_unknowns)
        arr_unknowns = cont.array_with_unknowns
        np.testing.assert_almost_equal(
            arr_unknowns[-1][1], correct_row_unknowns_no_zero)
        np.testing.assert_almost_equal(
            arr_unknowns[-1][0], correct_values_no_zero)

        # check if other match to what we get with __getitem__
        for v1, v2 in zip(arr_unknowns[:-1], cont):
            np.testing.assert_almost_equal(v1, v2)

    def test_mixedtype_metas(self):
        import Orange
        zoo = Orange.data.Table("zoo")
        dom = Orange.data.Domain(zoo.domain.attributes[2:], zoo.domain.class_var,
                                 zoo.domain.metas + zoo.domain.attributes[:2])
        t = zoo.transform(dom)
        cont = contingency.get_contingency(zoo, 2, t.domain.metas[1])
        assert_dist_equal(cont["1"], [38, 5])
        assert_dist_equal(cont, [[4, 54], [38, 5]])
        with zoo.unlocked():
            zoo[25][t.domain.metas[1]] = float("nan")
            zoo[0][2] = float("nan")
        cont = contingency.get_contingency(zoo, 2, t.domain.metas[1])
        assert_dist_equal(cont["1"], [37, 5])
        assert_dist_equal(cont, [[4, 53], [37, 5]])
        np.testing.assert_almost_equal(cont.col_unknowns, [0, 1])
        np.testing.assert_almost_equal(cont.row_unknowns, [0, 1])
        self.assertEqual(0, cont.unknowns)

    @staticmethod
    def _construct_sparse():
        domain = data.Domain(
            [data.DiscreteVariable("d%i" % i, values=tuple("abc"))
             for i in range(10)] +
            [data.ContinuousVariable("c%i" % i) for i in range(10)],
            data.DiscreteVariable("y", values=tuple("abc")))

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
        X.data = X.data.copy()  # make it the owner of it's data
        Y = np.array([[1, 2, 1, 0, 0]]).T
        return data.Table.from_numpy(domain, X, Y)

    def test_sparse(self):
        d = self._construct_sparse()
        cont = contingency.Discrete(d, 5)
        assert_dist_equal(cont[0], [2, 0, 0])
        assert_dist_equal(cont["b"], [0, 1, 1])
        assert_dist_equal(cont[2], [1, 0, 0])

        cont = contingency.Continuous(d, 14)
        assert_dist_equal(cont[0], [[], []])
        assert_dist_equal(cont["b"], [[1], [1]])
        assert_dist_equal(cont[2], [[2], [1]])

        cont = contingency.Continuous(d, "c3")
        assert_dist_equal(cont[0], [[1.1], [1]])
        assert_dist_equal(cont["b"], [[1], [1]])
        assert_dist_equal(cont[2], [[], []])

        with d.unlocked():
            d[4].set_class(1)
        cont = contingency.Continuous(d, 13)
        assert_dist_equal(cont[0], [[], []])
        assert_dist_equal(cont["b"], [[1, 1.1], [1, 1]])
        assert_dist_equal(cont[2], [[], []])

        cont = contingency.Continuous(d, 12)
        assert_dist_equal(cont[0], [[], []])
        assert_dist_equal(cont["b"], [[], []])
        assert_dist_equal(cont[2], [[], []])


    def test_get_contingency(self):
        d = self._construct_sparse()
        cont = contingency.get_contingency(d, 5)
        self.assertIsInstance(cont, contingency.Discrete)
        assert_dist_equal(cont[0], [2, 0, 0])
        assert_dist_equal(cont["b"], [0, 1, 1])
        assert_dist_equal(cont[2], [1, 0, 0])

        cont = contingency.get_contingency(d, "c4")
        self.assertIsInstance(cont, contingency.Continuous)
        assert_dist_equal(cont[0], [[], []])
        assert_dist_equal(cont["b"], [[1], [1]])
        assert_dist_equal(cont[2], [[2], [1]])

        cont = contingency.get_contingency(d, d.domain[13])
        self.assertIsInstance(cont, contingency.Continuous)
        assert_dist_equal(cont[0], [[1.1], [1]])
        assert_dist_equal(cont["b"], [[1], [1]])
        assert_dist_equal(cont[2], [[], []])
        assert_dist_equal(cont[2], [[], []])

    def test_get_contingencies(self):
        d = self._construct_sparse()
        conts = contingency.get_contingencies(d)

        self.assertEqual(len(conts), 20)

        cont = conts[5]
        self.assertIsInstance(cont, contingency.Discrete)
        assert_dist_equal(cont[0], [2, 0, 0])
        assert_dist_equal(cont["b"], [0, 1, 1])
        assert_dist_equal(cont[2], [1, 0, 0])

        cont = conts[14]
        self.assertIsInstance(cont, contingency.Continuous)
        assert_dist_equal(cont[0], [[], []])
        assert_dist_equal(cont["b"], [[1], [1]])
        assert_dist_equal(cont[2], [[2], [1]])

        conts = contingency.get_contingencies(d, skip_discrete=True)
        self.assertEqual(len(conts), 10)
        cont = conts[4]
        self.assertIsInstance(cont, contingency.Continuous)
        assert_dist_equal(cont[0], [[], []])
        assert_dist_equal(cont["b"], [[1], [1]])
        assert_dist_equal(cont[2], [[2], [1]])

        conts = contingency.get_contingencies(d, skip_continuous=True)
        self.assertEqual(len(conts), 10)
        cont = conts[5]
        self.assertIsInstance(cont, contingency.Discrete)
        assert_dist_equal(cont[0], [2, 0, 0])
        assert_dist_equal(cont["b"], [0, 1, 1])
        assert_dist_equal(cont[2], [1, 0, 0])

    def test_compute_contingency_metas(self):
        var1, var2 = self.test9.domain[-2], self.test9.domain[-4]
        cont = contingency.Discrete(self.test9, var1, var2)
        assert_dist_equal(cont, [[3, 0, 0], [0, 2, 0],
                                 [0, 0, 2], [0, 1, 0]])

    def test_compute_contingency_row_attribute_sparse(self):
        """
        Testing with sparse row variable since currently we do not test the
        situation when a row variable is sparse.
        """
        # make X sparse
        d = self.test9.copy()
        with d.unlocked():
            d.X = csr_matrix(d.X)
        var1, var2 = d.domain[0], d.domain[1]
        cont = contingency.Discrete(d, var1, var2)
        assert_dist_equal(cont, [[1, 0], [1, 0], [1, 0], [1, 0],
                                 [0, 1], [0, 1], [0, 1], [0, 1]])
        cont = contingency.Discrete(d, var2, var1)
        assert_dist_equal(cont, [[1, 1, 1, 1, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 1, 1, 1, 1]])

        d = self.test9.copy()
        with d.unlocked():
            d.X = csc_matrix(d.X)
        cont = contingency.Discrete(d, var1, var2)
        assert_dist_equal(cont, [[1, 0], [1, 0], [1, 0], [1, 0],
                                 [0, 1], [0, 1], [0, 1], [0, 1]])
        cont = contingency.Discrete(d, var2, var1)
        assert_dist_equal(cont, [[1, 1, 1, 1, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 1, 1, 1, 1]])

    def test_compute_contingency_invalid(self):
        rstate = np.random.RandomState(0xFFFF)
        X = data.ContinuousVariable("X")
        C = data.DiscreteVariable("C", values=["C{}".format(i + 1) for i in range(1024)])
        domain = data.Domain([X], [C])
        d = data.Table.from_numpy(
            domain,
            rstate.uniform(size=(20, 1)).round(1),
            rstate.randint(0, 1024, size=(20, 1)),
        )
        c = contingency.get_contingency(d, X, C)
        self.assertEqual(c.counts.shape[0], 1024)

        with d.unlocked():
            d.Y[5] = 1024
        with self.assertRaises(IndexError):
            contingency.get_contingency(d, X, C)

    def test_incompatible_arguments(self):
        """
        When providing data table unknowns should not be provided
        """
        self.assertRaises(
            TypeError, contingency.Discrete, self.zoo, 0, unknowns=0)
        self.assertRaises(
            TypeError, contingency.Discrete, self.zoo, 0, row_unknowns=0)
        self.assertRaises(
            TypeError, contingency.Discrete, self.zoo, 0, col_unknowns=0)

        self.assertRaises(
            TypeError, contingency.Continuous, self.zoo, 0, unknowns=0)
        self.assertRaises(
            TypeError, contingency.Continuous, self.zoo, 0, row_unknowns=0)
        self.assertRaises(
            TypeError, contingency.Continuous, self.zoo, 0, col_unknowns=0)

        # data with no class
        zoo_ = Table.from_table(Domain(self.zoo.domain.attributes), self.zoo)
        self.assertRaises(ValueError, contingency.Discrete, zoo_, 0)
        self.assertRaises(ValueError, contingency.Continuous, zoo_, 0)
