import unittest
import numpy as np
from scipy.sparse import csr_matrix

from Orange.statistics.util import bincount, countnans, contingency, stats


class TestUtil(unittest.TestCase):
    def test_bincount(self):
        hist, n_nans = bincount([0., 1., np.nan, 3])
        self.assertEqual(n_nans, 1)
        np.testing.assert_equal(hist, [1, 1, 0, 1])

        hist, n_nans = bincount([0., 1., 3], max_val=3)
        self.assertEqual(n_nans, 0)
        np.testing.assert_equal(hist, [1, 1, 0, 1])

    def test_countnans(self):
        np.testing.assert_equal(countnans([[1, np.nan],
                                           [2, np.nan]], axis=0), [0, 2])

    def test_contingency(self):
        x = np.array([0, 1, 0, 2, np.nan])
        y = np.array([0, 0, 1, np.nan, 0])
        cont_table, nans = contingency(x, y, 2, 2)
        np.testing.assert_equal(cont_table, [[1, 1, 0],
                                             [1, 0, 0],
                                             [0, 0, 0]])
        np.testing.assert_equal(nans, [1, 0, 0])

    def test_stats(self):
        X = np.arange(4).reshape(2, 2).astype(float)
        X[1, 1] = np.nan
        np.testing.assert_equal(stats(X), [[0, 2, 1, 0, 0, 2],
                                           [1, 1, 1, 0, 1, 1]])

    def test_stats_sparse(self):
        X = csr_matrix(np.identity(5))
        np.testing.assert_equal(stats(X), [[0, 1, .2, 0, 4, 1],
                                           [0, 1, .2, 0, 4, 1],
                                           [0, 1, .2, 0, 4, 1],
                                           [0, 1, .2, 0, 4, 1],
                                           [0, 1, .2, 0, 4, 1]])
