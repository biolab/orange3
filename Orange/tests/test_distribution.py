import unittest

import numpy as np
import scipy.sparse as sp

from Orange.statistics import distribution
from Orange import data


class Distribution_DiscreteTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        data.table.dataset_dirs.append("Orange/tests")

    def setUp(self):
        self.freqs = [4.0, 20.0, 13.0, 8.0, 10.0, 41.0, 5.0]
        s = sum(self.freqs)
        self.rfreqs = [x/s for x in self.freqs]

    def test_from_table(self):
        d = data.Table("zoo")
        disc = distribution.Discrete(d, "type")
        self.assertIsInstance(disc, np.ndarray)
        self.assertIs(disc.variable, d.domain["type"])
        self.assertEqual(disc.unknowns, 0)
        np.testing.assert_array_equal(disc, self.freqs)

        disc2 = distribution.Discrete(d, d.domain.class_var)
        self.assertIsInstance(disc2, np.ndarray)
        self.assertIs(disc2.variable, d.domain.class_var)
        self.assertEqual(disc, disc2)

        disc3 = distribution.Discrete(d, len(d.domain.attributes))
        self.assertIsInstance(disc3, np.ndarray)
        self.assertIs(disc3.variable, d.domain.class_var)
        self.assertEqual(disc, disc3)

        disc5 = distribution.class_distribution(d)
        self.assertIsInstance(disc5, np.ndarray)
        self.assertIs(disc5.variable, d.domain.class_var)
        self.assertEqual(disc, disc5)


    def test_construction(self):
        d = data.Table("zoo")

        disc = distribution.Discrete(d, "type")
        self.assertIsInstance(disc, np.ndarray)
        self.assertIs(disc.variable, d.domain["type"])
        self.assertEqual(disc.unknowns, 0)
        self.assertIs(disc.variable, d.domain.class_var)

        disc7 = distribution.Discrete(self.freqs)
        self.assertIsInstance(disc, np.ndarray)
        self.assertIsNone(disc7.variable)
        self.assertEqual(disc7.unknowns, 0)
        self.assertEqual(disc, disc7)

        disc1 = distribution.Discrete(None, d.domain.class_var)
        self.assertIsInstance(disc1, np.ndarray)
        self.assertIs(disc1.variable, d.domain.class_var)
        self.assertEqual(disc.unknowns, 0)
        np.testing.assert_array_equal(disc1,
                                      [0]*len(d.domain.class_var.values))


    def test_indexing(self):
        d = data.Table("zoo")
        indamphibian = d.domain.class_var.to_val("amphibian")

        disc = distribution.class_distribution(d)

        self.assertEqual(len(disc), len(d.domain.class_var.values))

        self.assertEqual(disc["mammal"], 41)
        self.assertEqual(disc[indamphibian], 4)

        disc["mammal"] = 100
        self.assertEqual(disc[d.domain.class_var.to_val("mammal")], 100)

        disc[indamphibian] = 33
        self.assertEqual(disc["amphibian"], 33)

        disc = distribution.class_distribution(d)
        self.assertEqual(list(disc), self.freqs)


    def test_hash(self):
        d = data.Table("zoo")
        disc = distribution.Discrete(d, "type")

        disc2 = distribution.Discrete(d, d.domain.class_var)
        self.assertEqual(hash(disc), hash(disc2))

        disc2[0] += 1
        self.assertNotEqual(hash(disc), hash(disc2))

        disc2[0] -= 1
        self.assertEqual(hash(disc), hash(disc2))

        disc2.unknowns += 1
        self.assertNotEqual(hash(disc), hash(disc2))


    def test_add(self):
        d = data.Table("zoo")
        disc = distribution.Discrete(d, "type")

        disc += [1,    2,     3,   4,    5,    6,   7]
        self.assertEqual(disc, [5.0, 22.0, 16.0, 12.0, 15.0, 47.0, 12.0])

        disc2 = distribution.Discrete(d, d.domain.class_var)

        disc3 = disc - disc2
        self.assertEqual(disc3, list(range(1, 8)))

        disc3 *= 2
        self.assertEqual(disc3, [2*x for x in range(1, 8)])


    def test_normalize(self):
        d = data.Table("zoo")
        disc = distribution.Discrete(d, "type")
        disc.normalize()
        self.assertEqual(disc, self.rfreqs)
        disc.normalize()
        self.assertEqual(disc, self.rfreqs)

        disc1 = distribution.Discrete(None, d.domain.class_var)
        disc1.normalize()
        v = len(d.domain.class_var.values)
        np.testing.assert_almost_equal(disc1, [1/v]*v)


    def test_modus(self):
        d = data.Table("zoo")
        disc = distribution.Discrete(d, "type")
        self.assertEqual(str(disc.modus()), "mammal")


    def test_random(self):
        d = data.Table("zoo")
        disc = distribution.Discrete(d, "type")
        ans = set()
        for i in range(1000):
            ans.add(int(disc.random()))
        self.assertEqual(ans, set(range(len(d.domain.class_var.values))))


class Distribution_ContinuousTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        data.table.dataset_dirs.append("Orange/tests")

    def setUp(self):
        self.freqs = np.array([(1.0, 1), (1.1, 1), (1.2, 2), (1.3, 7), (1.4, 12),
                      (1.5, 14), (1.6, 7), (1.7, 4), (1.9, 2), (3.0, 1),
                      (3.3, 2), (3.5, 2), (3.6, 1), (3.7, 1), (3.8, 1),
                      (3.9, 3), (4.0, 5), (4.1, 3), (4.2, 4), (4.3, 2),
                      (4.4, 4), (4.5, 8), (4.6, 3), (4.7, 5), (4.8, 4),
                      (4.9, 5), (5.0, 4), (5.1, 8), (5.2, 2), (5.3, 2),
                      (5.4, 2), (5.5, 3), (5.6, 6), (5.7, 3), (5.8, 3),
                      (5.9, 2), (6.0, 2), (6.1, 3), (6.3, 1), (6.4, 1),
                      (6.6, 1), (6.7, 2), (6.9, 1)]).T


    def test_from_table(self):
        d = data.Table("iris")
        petal_length = d.columns.petal_length

        disc = distribution.Continuous(d, "petal length")
        self.assertIsInstance(disc, np.ndarray)
        self.assertIs(disc.variable, petal_length)
        self.assertEqual(disc.unknowns, 0)
        np.testing.assert_almost_equal(disc, self.freqs)

        disc2 = distribution.Continuous(d, d.domain[2])
        self.assertIsInstance(disc2, np.ndarray)
        self.assertIs(disc2.variable, petal_length)
        self.assertEqual(disc, disc2)

        disc3 = distribution.Continuous(d, 2)
        self.assertIsInstance(disc3, np.ndarray)
        self.assertIs(disc3.variable, petal_length)
        self.assertEqual(disc, disc3)



    def test_construction(self):
        d = data.Table("iris")
        petal_length = d.columns.petal_length

        disc = distribution.Continuous(d, "petal length")

        disc7 = distribution.Continuous(self.freqs)
        self.assertIsInstance(disc, np.ndarray)
        self.assertIsNone(disc7.variable)
        self.assertEqual(disc7.unknowns, 0)
        self.assertEqual(disc, disc7)

        disc7 = distribution.Continuous(self.freqs, petal_length)
        self.assertIsInstance(disc, np.ndarray)
        self.assertIs(disc7.variable, petal_length)
        self.assertEqual(disc7.unknowns, 0)
        self.assertEqual(disc, disc7)

        disc1 = distribution.Continuous(10, petal_length)
        self.assertIsInstance(disc1, np.ndarray)
        self.assertIs(disc7.variable, petal_length)
        self.assertEqual(disc7.unknowns, 0)
        np.testing.assert_array_equal(disc1, np.zeros((2, 10)))

        dd = [list(range(5)), [1, 1, 2, 5, 1]]
        disc2 = distribution.Continuous(dd)
        self.assertIsInstance(disc2, np.ndarray)
        self.assertIsNone(disc2.variable)
        self.assertEqual(disc2.unknowns, 0)
        np.testing.assert_array_equal(disc2, dd)

    def test_hash(self):
        d = data.Table("iris")
        petal_length = d.columns.petal_length

        disc = distribution.Continuous(d, "petal length")
        disc2 = distribution.Continuous(d, petal_length)
        self.assertEqual(hash(disc), hash(disc2))

        disc2[0, 0] += 1
        self.assertNotEqual(hash(disc), hash(disc2))

        disc2[0, 0] -= 1
        self.assertEqual(hash(disc), hash(disc2))

        disc2.unknowns += 1
        self.assertNotEqual(hash(disc), hash(disc2))


    def test_normalize(self):
        d = data.Table("iris")
        petal_length = d.columns.petal_length

        disc = distribution.Continuous(d, "petal length")

        np.testing.assert_almost_equal(disc, self.freqs)
        disc.normalize()
        self.freqs[1, :] /= 150
        np.testing.assert_almost_equal(disc, self.freqs)

        disc1 = distribution.Continuous(10, petal_length)
        disc1.normalize()
        f = np.zeros((2, 10))
        f[1, :] = 0.1
        np.testing.assert_almost_equal(disc1, f)


    def test_modus(self):
        d = data.Table("iris")
        petal_length = d.columns.petal_length

        disc = distribution.Continuous([list(range(5)), [1, 1, 2, 5, 1]])
        self.assertEqual(disc.modus(), 3)


    def test_random(self):
        d = data.Table("iris")
        petal_length = d.columns.petal_length

        disc = distribution.Continuous(d, "petal length")
        ans = set()
        for i in range(1000):
            v = disc.random()
            self.assertTrue(v in self.freqs)
            ans.add(v)
        self.assertTrue(len(ans) > 10)


class Class_Distribution_Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        data.table.dataset_dirs.append("Orange/tests")

    def test_class_distribution(self):
        d = data.Table("zoo")
        disc = distribution.class_distribution(d)
        self.assertIsInstance(disc, np.ndarray)
        self.assertIs(disc.variable, d.domain["type"])
        self.assertEqual(disc.unknowns, 0)
        np.testing.assert_array_equal(disc,
                                      [4.0, 20.0, 13.0, 8.0, 10.0, 41.0, 5.0])

class Get_Distribution_Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        data.table.dataset_dirs.append("Orange/tests")

    def test_get_distribution(self):
        d = data.Table("iris")
        cls = d.domain.class_var
        disc = distribution.get_distribution(d, cls)
        self.assertIsInstance(disc, np.ndarray)
        self.assertIs(disc.variable, cls)
        self.assertEqual(disc.unknowns, 0)
        np.testing.assert_array_equal(disc, [50, 50, 50])

        petal_length = d.columns.petal_length
        freqs = np.array([(1.0, 1), (1.1, 1), (1.2, 2), (1.3, 7), (1.4, 12),
                      (1.5, 14), (1.6, 7), (1.7, 4), (1.9, 2), (3.0, 1),
                      (3.3, 2), (3.5, 2), (3.6, 1), (3.7, 1), (3.8, 1),
                      (3.9, 3), (4.0, 5), (4.1, 3), (4.2, 4), (4.3, 2),
                      (4.4, 4), (4.5, 8), (4.6, 3), (4.7, 5), (4.8, 4),
                      (4.9, 5), (5.0, 4), (5.1, 8), (5.2, 2), (5.3, 2),
                      (5.4, 2), (5.5, 3), (5.6, 6), (5.7, 3), (5.8, 3),
                      (5.9, 2), (6.0, 2), (6.1, 3), (6.3, 1), (6.4, 1),
                      (6.6, 1), (6.7, 2), (6.9, 1)]).T
        disc = distribution.get_distribution(d, petal_length)
        np.testing.assert_almost_equal(disc, freqs)

class Domain_Distribution_Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        data.table.dataset_dirs.append("Orange/tests")

    def test_get_distributions(self):
        d = data.Table("iris")
        ddist = distribution.get_distributions(d)

        self.assertEqual(len(ddist), 5)
        for i in range(4):
            self.assertIsInstance(ddist[i], distribution.Continuous)
        self.assertIsInstance(ddist[-1], distribution.Discrete)

        freqs = np.array([(1.0, 1), (1.1, 1), (1.2, 2), (1.3, 7), (1.4, 12),
                      (1.5, 14), (1.6, 7), (1.7, 4), (1.9, 2), (3.0, 1),
                      (3.3, 2), (3.5, 2), (3.6, 1), (3.7, 1), (3.8, 1),
                      (3.9, 3), (4.0, 5), (4.1, 3), (4.2, 4), (4.3, 2),
                      (4.4, 4), (4.5, 8), (4.6, 3), (4.7, 5), (4.8, 4),
                      (4.9, 5), (5.0, 4), (5.1, 8), (5.2, 2), (5.3, 2),
                      (5.4, 2), (5.5, 3), (5.6, 6), (5.7, 3), (5.8, 3),
                      (5.9, 2), (6.0, 2), (6.1, 3), (6.3, 1), (6.4, 1),
                      (6.6, 1), (6.7, 2), (6.9, 1)]).T
        np.testing.assert_almost_equal(ddist[2], freqs)
        np.testing.assert_almost_equal(ddist[-1], [50, 50, 50])


    #noinspection PyTypeChecker
    def test_sparse_get_distributions(self):
        domain = data.Domain(
            [data.DiscreteVariable("d%i" % i, values=list("abc"))
                 for i in range(10)] +
            [data.ContinuousVariable("c%i" % i) for i in range(10)])

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
        d = data.Table.from_numpy(domain, X)

        ddist = distribution.get_distributions(d)

        self.assertEqual(len(ddist), 20)
        np.testing.assert_almost_equal(ddist[0], [0, 0, 0])
        np.testing.assert_almost_equal(ddist[1], [0, 0, 1])
        np.testing.assert_almost_equal(ddist[2], [0, 1, 1])
        np.testing.assert_almost_equal(ddist[3], [0, 2, 1])
        np.testing.assert_almost_equal(ddist[4], [1, 1, 0])
        np.testing.assert_almost_equal(ddist[5], [2, 1, 1])
        np.testing.assert_almost_equal(ddist[6], [1, 2, 1])
        np.testing.assert_almost_equal(ddist[7], [0, 0, 0])
        np.testing.assert_almost_equal(ddist[8], [0, 0, 1])
        np.testing.assert_almost_equal(ddist[9], [0, 1, 0])

        z = np.zeros((2, 0))
        np.testing.assert_almost_equal(ddist[10], z)
        np.testing.assert_almost_equal(ddist[11], z)
        np.testing.assert_almost_equal(ddist[12], z)
        np.testing.assert_almost_equal(ddist[13], [[1, 1.1], [1, 1]])
        np.testing.assert_almost_equal(ddist[14], [[1, 2], [1, 1]])
        np.testing.assert_almost_equal(ddist[15], z)
        np.testing.assert_almost_equal(ddist[16], [[1, 2], [1, 1]])
        np.testing.assert_almost_equal(ddist[17], [[0], [2]])
        np.testing.assert_almost_equal(ddist[18], [[2], [1]])
        np.testing.assert_almost_equal(ddist[19], z)


        d.set_weights(np.array([1, 2, 3, 4, 5]))

        ddist = distribution.get_distributions(d)

        self.assertEqual(len(ddist), 20)
        np.testing.assert_almost_equal(ddist[0], [0, 0, 0])
        np.testing.assert_almost_equal(ddist[1], [0, 0, 1])
        np.testing.assert_almost_equal(ddist[2], [0, 2, 5])
        np.testing.assert_almost_equal(ddist[3], [0, 5, 1])
        np.testing.assert_almost_equal(ddist[4], [2, 1, 0])
        np.testing.assert_almost_equal(ddist[5], [7, 1, 3])
        np.testing.assert_almost_equal(ddist[6], [3, 7, 1])
        np.testing.assert_almost_equal(ddist[7], [0, 0, 0])
        np.testing.assert_almost_equal(ddist[8], [0, 0, 2])
        np.testing.assert_almost_equal(ddist[9], [0, 1, 0])

        z = np.zeros((2, 0))
        np.testing.assert_almost_equal(ddist[10], z)
        np.testing.assert_almost_equal(ddist[11], z)
        np.testing.assert_almost_equal(ddist[12], z)
        np.testing.assert_almost_equal(ddist[13], [[1, 1.1], [1, 5]])
        np.testing.assert_almost_equal(ddist[14], [[1, 2], [1, 2]])
        np.testing.assert_almost_equal(ddist[15], z)
        np.testing.assert_almost_equal(ddist[16], [[1, 2], [2, 1]])
        np.testing.assert_almost_equal(ddist[17], [[0], [3]])
        np.testing.assert_almost_equal(ddist[18], [[2], [1]])
        np.testing.assert_almost_equal(ddist[19], z)

if __name__ == "__main__":
    unittest.main()
