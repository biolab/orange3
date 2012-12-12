import unittest

import numpy as np
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
        disc = distribution.Discrete("type", d)
        self.assertIsInstance(disc, np.ndarray)
        self.assertIs(disc.variable, d.domain["type"])
        self.assertEqual(disc.unknowns, 0)
        np.testing.assert_array_equal(disc, self.freqs)

        disc2 = distribution.Discrete(d.domain.class_var, d)
        self.assertIsInstance(disc2, np.ndarray)
        self.assertIs(disc2.variable, d.domain.class_var)
        self.assertEqual(disc, disc2)

        disc3 = distribution.Discrete(len(d.domain.attributes), d)
        self.assertIsInstance(disc3, np.ndarray)
        self.assertIs(disc3.variable, d.domain.class_var)
        self.assertEqual(disc, disc3)

        disc5 = distribution.class_distribution(d)
        self.assertIsInstance(disc5, np.ndarray)
        self.assertIs(disc5.variable, d.domain.class_var)
        self.assertEqual(disc, disc5)


    def test_construction(self):
        d = data.Table("zoo")

        disc = distribution.Discrete("type", d)
        self.assertIsInstance(disc, np.ndarray)
        self.assertIs(disc.variable, d.domain["type"])
        self.assertEqual(disc.unknowns, 0)
        self.assertIs(disc.variable, d.domain.class_var)

        disc7 = distribution.Discrete(None, self.freqs)
        self.assertIsInstance(disc, np.ndarray)
        self.assertIsNone(disc7.variable)
        self.assertEqual(disc7.unknowns, 0)
        self.assertEqual(disc, disc7)

        disc1 = distribution.Discrete(d.domain.class_var)
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
        disc = distribution.Discrete("type", d)

        disc2 = distribution.Discrete(d.domain.class_var, d)
        self.assertEqual(hash(disc), hash(disc2))

        disc2[0] += 1
        self.assertNotEqual(hash(disc), hash(disc2))

        disc2[0] -= 1
        self.assertEqual(hash(disc), hash(disc2))

        disc2.unknowns += 1
        self.assertNotEqual(hash(disc), hash(disc2))


    def test_add(self):
        d = data.Table("zoo")
        disc = distribution.Discrete("type", d)

        disc += [1,    2,     3,   4,    5,    6,   7]
        self.assertEqual(disc, [5.0, 22.0, 16.0, 12.0, 15.0, 47.0, 12.0])

        disc2 = distribution.Discrete(d.domain.class_var, d)

        disc3 = disc - disc2
        self.assertEqual(disc3, list(range(1, 8)))

        disc3 *= 2
        self.assertEqual(disc3, [2*x for x in range(1, 8)])


    def test_normalize(self):
        d = data.Table("zoo")
        disc = distribution.Discrete("type", d)
        disc.normalize()
        self.assertEqual(disc, self.rfreqs)
        disc.normalize()
        self.assertEqual(disc, self.rfreqs)

        disc1 = distribution.Discrete(d.domain.class_var)
        disc1.normalize()
        v = len(d.domain.class_var.values)
        np.testing.assert_almost_equal(disc1, [1/v]*v)


    def test_modus(self):
        d = data.Table("zoo")
        disc = distribution.Discrete("type", d)
        self.assertEqual(str(disc.modus()), "mammal")


    def test_random(self):
        d = data.Table("zoo")
        disc = distribution.Discrete("type", d)
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

        disc = distribution.Continuous("petal length", d)
        self.assertIsInstance(disc, np.ndarray)
        self.assertIs(disc.variable, petal_length)
        self.assertEqual(disc.unknowns, 0)
        np.testing.assert_almost_equal(disc, self.freqs)

        disc2 = distribution.Continuous(d.domain[2], d)
        self.assertIsInstance(disc2, np.ndarray)
        self.assertIs(disc2.variable, petal_length)
        self.assertEqual(disc, disc2)

        disc3 = distribution.Continuous(2, d)
        self.assertIsInstance(disc3, np.ndarray)
        self.assertIs(disc3.variable, petal_length)
        self.assertEqual(disc, disc3)



    def test_construction(self):
        d = data.Table("iris")
        petal_length = d.columns.petal_length

        disc = distribution.Continuous("petal length", d)

        disc7 = distribution.Continuous(None, self.freqs)
        self.assertIsInstance(disc, np.ndarray)
        self.assertIsNone(disc7.variable)
        self.assertEqual(disc7.unknowns, 0)
        self.assertEqual(disc, disc7)

        disc7 = distribution.Continuous(petal_length, self.freqs)
        self.assertIsInstance(disc, np.ndarray)
        self.assertIs(disc7.variable, petal_length)
        self.assertEqual(disc7.unknowns, 0)
        self.assertEqual(disc, disc7)

        disc1 = distribution.Continuous(petal_length, 10)
        self.assertIsInstance(disc1, np.ndarray)
        self.assertIs(disc7.variable, petal_length)
        self.assertEqual(disc7.unknowns, 0)
        np.testing.assert_array_equal(disc1, np.zeros((2, 10)))

        dd = [list(range(5)), [1, 1, 2, 5, 1]]
        disc2 = distribution.Continuous(None, dd)
        self.assertIsInstance(disc2, np.ndarray)
        self.assertIsNone(disc2.variable)
        self.assertEqual(disc2.unknowns, 0)
        np.testing.assert_array_equal(disc2, dd)

    def test_hash(self):
        d = data.Table("iris")
        petal_length = d.columns.petal_length

        disc = distribution.Continuous("petal length", d)
        disc2 = distribution.Continuous(petal_length, d)
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

        disc = distribution.Continuous("petal length", d)

        np.testing.assert_almost_equal(disc, self.freqs)
        disc.normalize()
        self.freqs[1, :] /= 150
        np.testing.assert_almost_equal(disc, self.freqs)

        disc1 = distribution.Continuous(petal_length, 10)
        disc1.normalize()
        f = np.zeros((2, 10))
        f[1, :] = 0.1
        np.testing.assert_almost_equal(disc1, f)


    def test_modus(self):
        d = data.Table("iris")
        petal_length = d.columns.petal_length

        disc = distribution.Continuous(None, [list(range(5)), [1, 1, 2, 5, 1]])
        self.assertEqual(disc.modus(), 3)


    def test_random(self):
        d = data.Table("iris")
        petal_length = d.columns.petal_length

        disc = distribution.Continuous("petal length", d)
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
        disc = distribution.get_distribution(cls, d)
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
        disc = distribution.get_distribution(petal_length, d)
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

if __name__ == "__main__":
    unittest.main()