import unittest

import numpy as np

from Orange.data.util import one_hot

class TestDataUtil(unittest.TestCase):
    def test_one_hot(self):
        np.testing.assert_equal(
            one_hot([0, 1, 2, 1], int), [[1, 0, 0],
                                         [0, 1, 0],
                                         [0, 0, 1],
                                         [0, 1, 0]])
