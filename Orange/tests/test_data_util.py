import unittest

import numpy as np

from Orange.data.util import scale, one_hot

class TestDataUtil(unittest.TestCase):
    def test_scale(self):
        np.testing.assert_equal(scale([0, 1, 2], -1, 1), [-1, 0, 1])
        np.testing.assert_equal(scale([3, 3, 3]), [1, 1, 1])
        np.testing.assert_equal(scale([.1, .5, np.nan]), [0, 1, np.nan])

    def test_one_hot(self):
        np.testing.assert_equal(
            one_hot([0, 1, 2, 1], int), [[1, 0, 0],
                                         [0, 1, 0],
                                         [0, 0, 1],
                                         [0, 1, 0]])
