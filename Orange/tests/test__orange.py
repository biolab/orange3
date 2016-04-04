import unittest
import numpy as np
from Orange.data import _valuecount


class test_valuecount(unittest.TestCase):

    def test_valuecount(self):
        for a, expected_b in ([[[1, 1, 1, 1], [0.1, 0.2, 0.3, 0.4]], [[1], [1]]],
                              [[[1, 1, 1, 2], [0.1, 0.2, 0.3, 0.4]], [[1, 2], [0.6, 0.4]]],
                              [[[0, 1, 1, 1], [0.1, 0.2, 0.3, 0.4]], [[0, 1], [0.1, 0.9]]],
                              [[[0, 1, 1, 2], [0.1, 0.2, 0.3, 0.4]], [[0, 1, 2], [0.1, 0.5, 0.4]]],
                              [[[0, 1, 2, 3], [0.1, 0.2, 0.3, 0.4]], None],
                              [[[0], [0.1]], None],
                              [np.ones((2, 1)), None]):
            a = np.array(a)
            b = _valuecount.valuecount(a)
            if expected_b is not None:
                np.testing.assert_almost_equal(b, expected_b)
            else:
                np.testing.assert_almost_equal(b, a)

        for value in ([np.array([[0, 1], [2, 3]])],
                      [np.ones(2)],
                      [np.ones((3, 3))],
                      None):
            self.assertRaises(TypeError, _valuecount.valuecount, value)
