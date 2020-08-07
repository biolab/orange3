import unittest

from Orange.data import DiscreteVariable, ContinuousVariable
from Orange.preprocess.impute import ReplaceUnknownsRandom, ReplaceUnknowns
from Orange.statistics.distribution import Discrete


class TestReplaceUnknowns(unittest.TestCase):
    def test_equality(self):
        v1 = ContinuousVariable("x")
        v2 = ContinuousVariable("x")
        v3 = ContinuousVariable("y")

        t1 = ReplaceUnknowns(v1, 0)
        t1a = ReplaceUnknowns(v2, 0)
        t2 = ReplaceUnknowns(v3, 0)
        self.assertEqual(t1, t1)
        self.assertEqual(t1, t1a)
        self.assertNotEqual(t1, t2)

        self.assertEqual(hash(t1), hash(t1a))
        self.assertNotEqual(hash(t1), hash(t2))

        t1 = ReplaceUnknowns(v1, 0)
        t1a = ReplaceUnknowns(v1, 1)
        self.assertNotEqual(t1, t1a)
        self.assertNotEqual(hash(t1), hash(t1a))


class TestReplaceUnknownsRandom(unittest.TestCase):
    def test_equality(self):
        v1 = DiscreteVariable("x", tuple("abc"))
        v2 = DiscreteVariable("x", tuple("abc"))
        v3 = DiscreteVariable("y", tuple("abc"))

        d1 = Discrete([1, 2, 3], v1)
        d2 = Discrete([1, 2, 3], v2)
        d3 = Discrete([1, 2, 3], v3)

        t1 = ReplaceUnknownsRandom(v1, d1)
        t1a = ReplaceUnknownsRandom(v2, d2)
        t2 = ReplaceUnknownsRandom(v3, d3)
        self.assertEqual(t1, t1)
        self.assertEqual(t1, t1a)
        self.assertNotEqual(t1, t2)

        self.assertEqual(hash(t1), hash(t1a))
        self.assertNotEqual(hash(t1), hash(t2))

        d1[1] += 1
        self.assertNotEqual(t1, t1a)
        self.assertNotEqual(hash(t1), hash(t1a))


if __name__ == "__main__":
    unittest.main()
