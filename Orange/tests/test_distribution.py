import unittest

import numpy as np
from Orange.statistics import distribution
from Orange import data

class Distribution_DiscreteTestCase(unittest.TestCase):
    def setUp(self):
        self.freqs = [4.0, 20.0, 13.0, 8.0, 10.0, 41.0, 5.0]
        s = sum(self.freqs)
        self.rfreqs = [x/s for x in self.freqs]


    def test_from_table(self):
        d = data.Table("zoo")
        disc = distribution.Discrete("type", d)
        self.assertIsInstance(disc, np.ndarray)
        self.assertIs(disc.variable, d.domain["type"])
        self.assertEquals(disc.unknowns, 0)
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
        self.assertEquals(disc.unknowns, 0)
        self.assertIs(disc.variable, d.domain.class_var)

        disc7 = distribution.Discrete(None, self.freqs)
        self.assertIsInstance(disc, np.ndarray)
        self.assertIsNone(disc7.variable)
        self.assertEquals(disc7.unknowns, 0)
        self.assertEqual(disc, disc7)

        disc1 = distribution.Discrete(d.domain.class_var)
        self.assertIsInstance(disc1, np.ndarray)
        self.assertIs(disc1.variable, d.domain.class_var)
        self.assertEquals(disc.unknowns, 0)
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
