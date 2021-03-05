import unittest

import numpy as np

from Orange.data import \
    Domain, Table, \
    DiscreteVariable, ContinuousVariable, TimeVariable, StringVariable
from Orange.preprocess.impute import ReplaceUnknownsRandom, ReplaceUnknowns, \
    FixedValueByType
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


class TestFixedValuesByType(unittest.TestCase):
    def setUp(self):
        domain = Domain(
            [DiscreteVariable("d", values=tuple("abc")),
             ContinuousVariable("c"),
             TimeVariable("t")],
            [],
            [StringVariable("s")]
        )
        n = np.nan
        self.data = Table(
            domain,
            np.array([[1, n, 15], [n, 42, n]]),
            np.empty((2, 0)),
            np.array([["foo"], [""]]))

    def test_none_defined(self):
        d, c, t = self.data.domain.attributes
        s,  = self.data.domain.metas

        imputer = FixedValueByType()
        for var in (d, c, t):
            imp = imputer(self.data, var)
            self.assertIsInstance(imp.compute_value, ReplaceUnknowns)
            self.assertTrue(np.isnan(imp.compute_value.value))
        imp = imputer(self.data, s)
        self.assertIsInstance(imp.compute_value, ReplaceUnknowns)
        self.assertIsNone(imp.compute_value.value)

    def test_all_defined(self):
        d, c, t = self.data.domain.attributes
        s, = self.data.domain.metas

        imputer = FixedValueByType(
            default_discrete=1, default_continuous=42,
            default_string="foo", default_time=3.14)

        self.assertEqual(imputer(self.data, d).compute_value.value, 1)
        self.assertEqual(imputer(self.data, c).compute_value.value, 42)
        self.assertEqual(imputer(self.data, t).compute_value.value, 3.14)
        self.assertEqual(imputer(self.data, s).compute_value.value, "foo")

    def test_with_default(self):
        s, = self.data.domain.metas

        imputer = FixedValueByType(
            default_discrete=1, default_continuous=42,
            default_string="foo", default_time=3.14)

        self.assertEqual(
            imputer(self.data, s, default="bar").compute_value.value,
            "bar")


if __name__ == "__main__":
    unittest.main()
