import unittest
import numpy as np
from Orange.data import _valuecount


class test_valuecount(unittest.TestCase):
    def test_valuecount(self):
        a = np.array([[1, 1, 1, 1], [0.1, 0.2, 0.3, 0.4]])
        b = _valuecount.valuecount(a)
        np.testing.assert_almost_equal(b, [[1], [1]])

        a = np.array([[1, 1, 1, 2], [0.1, 0.2, 0.3, 0.4]])
        b = _valuecount.valuecount(a)
        np.testing.assert_almost_equal(b, [[1, 2], [0.6, 0.4]])

        a = np.array([[0, 1, 1, 1], [0.1, 0.2, 0.3, 0.4]])
        b = _valuecount.valuecount(a)
        np.testing.assert_almost_equal(b, [[0, 1], [0.1, 0.9]])

        a = np.array([[0, 1, 1, 2], [0.1, 0.2, 0.3, 0.4]])
        b = _valuecount.valuecount(a)
        np.testing.assert_almost_equal(b, [[0, 1, 2], [0.1, 0.5, 0.4]])

        a = np.array([[0, 1, 2, 3], [0.1, 0.2, 0.3, 0.4]])
        b = _valuecount.valuecount(a)
        np.testing.assert_almost_equal(b, a)

        a = np.array([[0], [0.1]])
        b = _valuecount.valuecount(a)
        np.testing.assert_almost_equal(b, a)

        a = np.ones((2, 1))
        b = _valuecount.valuecount(a)
        np.testing.assert_almost_equal(b, a)

        a = np.array([[0, 1], [2, 3]])
        self.assertRaises(ValueError, _valuecount.valuecount, a)

        a = np.ones(2)
        self.assertRaises(ValueError, _valuecount.valuecount, a)

        a = np.ones((3, 3))
        self.assertRaises(ValueError, _valuecount.valuecount, a)

        self.assertRaises(TypeError, _valuecount.valuecount, None)
