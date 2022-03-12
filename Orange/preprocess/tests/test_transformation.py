import unittest

import numpy as np
import scipy.sparse as sp

from Orange.data import DiscreteVariable
from Orange.preprocess.transformation import \
    Transformation, _Indicator, Normalizer, Lookup, Indicator, Indicator1


class TestTransformEquality(unittest.TestCase):
    def setUp(self):
        self.disc1 = DiscreteVariable("d1", values=tuple("abc"))
        self.disc1a = DiscreteVariable("d1", values=tuple("abc"))
        self.disc2 = DiscreteVariable("d2", values=tuple("abc"))
        assert self.disc1 == self.disc1a

    def test_transformation(self):
        t1 = Transformation(self.disc1)
        t1a = Transformation(self.disc1a)
        t2 = Transformation(self.disc2)
        self.assertEqual(t1, t1)
        self.assertEqual(t1, t1a)
        self.assertNotEqual(t1, t2)

        self.assertEqual(hash(t1), hash(t1a))
        self.assertNotEqual(hash(t1), hash(t2))

    def test_indicator(self):
        t1 = _Indicator(self.disc1, 0)
        t1a = _Indicator(self.disc1a, 0)
        t2 = _Indicator(self.disc2, 0)
        self.assertEqual(t1, t1)
        self.assertEqual(t1, t1a)
        self.assertNotEqual(t1, t2)

        self.assertEqual(hash(t1), hash(t1a))
        self.assertNotEqual(hash(t1), hash(t2))

        t1 = _Indicator(self.disc1, 0)
        t1a = _Indicator(self.disc1a, 1)
        self.assertNotEqual(t1, t1a)
        self.assertNotEqual(hash(t1), hash(t1a))

    def test_normalizer(self):
        t1 = Normalizer(self.disc1, 0, 1)
        t1a = Normalizer(self.disc1a, 0, 1)
        t2 = Normalizer(self.disc2, 0, 1)
        self.assertEqual(t1, t1)
        self.assertEqual(t1, t1a)
        self.assertNotEqual(t1, t2)

        self.assertEqual(hash(t1), hash(t1a))
        self.assertNotEqual(hash(t1), hash(t2))

        t1 = Normalizer(self.disc1, 0, 1)
        t1a = Normalizer(self.disc1a, 1, 1)
        self.assertNotEqual(t1, t1a)
        self.assertNotEqual(hash(t1), hash(t1a))

        t1 = Normalizer(self.disc1, 0, 1)
        t1a = Normalizer(self.disc1a, 0, 2)
        self.assertNotEqual(t1, t1a)
        self.assertNotEqual(hash(t1), hash(t1a))

    def test_lookup(self):
        t1 = Lookup(self.disc1, np.array([0, 2, 1]), 1)
        t1a = Lookup(self.disc1a, np.array([0, 2, 1]), 1)
        t2 = Lookup(self.disc2, np.array([0, 2, 1]), 1)
        self.assertEqual(t1, t1)
        self.assertEqual(t1, t1a)
        self.assertNotEqual(t1, t2)

        self.assertEqual(hash(t1), hash(t1a))
        self.assertNotEqual(hash(t1), hash(t2))

        t1 = Lookup(self.disc1, np.array([0, 2, 1]), 1)
        t1a = Lookup(self.disc1a, np.array([1, 2, 0]), 1)
        self.assertNotEqual(t1, t1a)
        self.assertNotEqual(hash(t1), hash(t1a))

        t1 = Lookup(self.disc1, np.array([0, 2, 1]), 1)
        t1a = Lookup(self.disc1a, np.array([0, 2, 1]), 2)
        self.assertNotEqual(t1, t1a)
        self.assertNotEqual(hash(t1), hash(t1a))


class TestIndicator(unittest.TestCase):
    def test_nan(self):
        var = DiscreteVariable("d", tuple("abcde"))

        col = np.array([1.0, 4, 2, np.nan, 2, 0])

        transform = Indicator(var, 2).transform
        expected = [0, 0, 1, np.nan, 1, 0]
        np.testing.assert_equal(transform(col), expected)
        sparse = transform(sp.csr_matrix(col))
        self.assertTrue(sp.issparse(sparse))
        np.testing.assert_equal(sparse.toarray().ravel(), expected)
        self.assertEqual(transform(1), 0)
        self.assertEqual(transform(2), 1)
        self.assertTrue(np.isnan(transform(np.nan)))

        transform = Indicator(var, 0).transform
        expected = [0, 0, 0, np.nan, 0, 1]
        np.testing.assert_equal(transform(col), expected)
        sparse = transform(sp.csr_matrix(col))
        # Currently, this always returns dense array
        assert not sp.issparse(sparse)
        np.testing.assert_equal(sparse, expected)
        self.assertEqual(transform(1), 0)
        self.assertEqual(transform(0), 1)
        self.assertTrue(np.isnan(transform(np.nan)))

        transform = Indicator1(var, 2).transform
        expected = [-1, -1, 1, np.nan, 1, -1]
        np.testing.assert_equal(transform(col), expected)
        np.testing.assert_equal(transform(sp.csr_matrix(col).toarray().ravel()),
                                expected)
        self.assertEqual(transform(1), -1)
        self.assertEqual(transform(2), 1)
        self.assertTrue(np.isnan(transform(np.nan)))


if __name__ == '__main__':
    unittest.main()
