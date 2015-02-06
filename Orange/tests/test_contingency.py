import unittest

import numpy as np
import scipy.sparse as sp

from Orange.statistics import contingency
from Orange import data


class Discrete_Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        data.table.dataset_dirs.append("Orange/tests")

    def test_discrete(self):
        d = data.Table("zoo")
        cont = contingency.Discrete(d, 0)
        np.testing.assert_almost_equal(cont["amphibian"], [4, 0])
        np.testing.assert_almost_equal(cont,
            [[4, 0], [20, 0], [13, 0], [4, 4], [10, 0], [2, 39], [5, 0]])

        cont = contingency.Discrete(d, "predator")
        np.testing.assert_almost_equal(cont["fish"], [4, 9])
        np.testing.assert_almost_equal(cont,
            [[1, 3], [11, 9], [4, 9], [7, 1], [2, 8], [19, 22], [1, 4]])

        cont = contingency.Discrete(d, d.domain["predator"])
        np.testing.assert_almost_equal(cont["fish"], [4, 9])
        np.testing.assert_almost_equal(cont,
            [[1, 3], [11, 9], [4, 9], [7, 1], [2, 8], [19, 22], [1, 4]])


    def test_continuous(self):
        d = data.Table("iris")
        cont = contingency.Continuous(d, "sepal width")
        correct = [[2.3, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7,
                    3.8, 3.9, 4.0, 4.1, 4.2, 4.4],
                   [1, 1, 6, 5, 5, 2, 9, 6, 2, 3, 4, 2, 1, 1, 1, 1]]
        np.testing.assert_almost_equal(cont["Iris-setosa"], correct)

        correct = [[2.2, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.6, 3.8],
                   [1, 4, 2, 4, 8, 2, 12, 4, 5, 3, 2, 1, 2]]
        np.testing.assert_almost_equal(cont[d.domain.class_var.values.index("Iris-virginica")], correct)


    @staticmethod
    def _construct_sparse():
        domain = data.Domain(
            [data.DiscreteVariable("d%i" % i, values=list("abc"))
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
        Y = np.array([[1, 2, 1, 0, 0]]).T
        return data.Table.from_numpy(domain, X, Y)


    def test_sparse(self):
        d = self._construct_sparse()
        cont = contingency.Discrete(d, 5)
        np.testing.assert_almost_equal(cont[0], [1, 0, 0])
        np.testing.assert_almost_equal(cont["b"], [0, 1, 1])
        np.testing.assert_almost_equal(cont[2], [1, 0, 0])

        cont = contingency.Continuous(d, 14)
        np.testing.assert_almost_equal(cont[0], [[], []])
        np.testing.assert_almost_equal(cont["b"], [[1], [1]])
        np.testing.assert_almost_equal(cont[2], [[2], [1]])

        cont = contingency.Continuous(d, "c3")
        np.testing.assert_almost_equal(cont[0], [[1.1], [1]])
        np.testing.assert_almost_equal(cont["b"], [[1], [1]])
        np.testing.assert_almost_equal(cont[2], [[], []])

        d[4].set_class(1)
        cont = contingency.Continuous(d, 13)
        np.testing.assert_almost_equal(cont[0], [[], []])
        np.testing.assert_almost_equal(cont["b"], [[1, 1.1], [1, 1]])
        np.testing.assert_almost_equal(cont[2], [[], []])

        cont = contingency.Continuous(d, 12)
        np.testing.assert_almost_equal(cont[0], [[], []])
        np.testing.assert_almost_equal(cont["b"], [[], []])
        np.testing.assert_almost_equal(cont[2], [[], []])


    def test_get_contingency(self):
        d = self._construct_sparse()
        cont = contingency.get_contingency(d, 5)
        self.assertIsInstance(cont, contingency.Discrete)
        np.testing.assert_almost_equal(cont[0], [1, 0, 0])
        np.testing.assert_almost_equal(cont["b"], [0, 1, 1])
        np.testing.assert_almost_equal(cont[2], [1, 0, 0])

        cont = contingency.get_contingency(d, "c4")
        self.assertIsInstance(cont, contingency.Continuous)
        np.testing.assert_almost_equal(cont[0], [[], []])
        np.testing.assert_almost_equal(cont["b"], [[1], [1]])
        np.testing.assert_almost_equal(cont[2], [[2], [1]])

        cont = contingency.get_contingency(d, d.domain[13])
        self.assertIsInstance(cont, contingency.Continuous)
        np.testing.assert_almost_equal(cont[0], [[1.1], [1]])
        np.testing.assert_almost_equal(cont["b"], [[1], [1]])
        np.testing.assert_almost_equal(cont[2], [[], []])
        np.testing.assert_almost_equal(cont[2], [[], []])

    def test_get_contingencies(self):
        d = self._construct_sparse()
        conts = contingency.get_contingencies(d)

        self.assertEqual(len(conts), 20)

        cont = conts[5]
        self.assertIsInstance(cont, contingency.Discrete)
        np.testing.assert_almost_equal(cont[0], [1, 0, 0])
        np.testing.assert_almost_equal(cont["b"], [0, 1, 1])
        np.testing.assert_almost_equal(cont[2], [1, 0, 0])

        cont = conts[14]
        self.assertIsInstance(cont, contingency.Continuous)
        np.testing.assert_almost_equal(cont[0], [[], []])
        np.testing.assert_almost_equal(cont["b"], [[1], [1]])
        np.testing.assert_almost_equal(cont[2], [[2], [1]])

        conts = contingency.get_contingencies(d, skipDiscrete=True)
        self.assertEqual(len(conts), 10)
        cont = conts[4]
        self.assertIsInstance(cont, contingency.Continuous)
        np.testing.assert_almost_equal(cont[0], [[], []])
        np.testing.assert_almost_equal(cont["b"], [[1], [1]])
        np.testing.assert_almost_equal(cont[2], [[2], [1]])

        conts = contingency.get_contingencies(d, skipContinuous=True)
        self.assertEqual(len(conts), 10)
        cont = conts[5]
        self.assertIsInstance(cont, contingency.Discrete)
        np.testing.assert_almost_equal(cont[0], [1, 0, 0])
        np.testing.assert_almost_equal(cont["b"], [0, 1, 1])
        np.testing.assert_almost_equal(cont[2], [1, 0, 0])
