# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring

import unittest
import numpy as np

from Orange.data import Table
from Orange.clustering.louvain import Louvain


class TestSVMLearner(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = Table('iris')
        cls.louvain = Louvain()

    def test_orange_table(self):
        self.assertIsNone(self.louvain.fit(self.data))
        clusters = self.louvain.fit_predict(self.data)
        self.assertIn(type(clusters), [list, np.ndarray])

    def test_np_array(self):
        data_np = self.data.X
        self.assertIsNone(self.louvain.fit(data_np))
        clusters = self.louvain.fit_predict(data_np)
        self.assertIn(type(clusters), [list, np.ndarray])
