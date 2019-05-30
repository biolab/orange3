# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring

import unittest
import numpy as np

from Orange.data import Table
from Orange.clustering.louvain import Louvain


class TestLouvain(unittest.TestCase):
    def setUp(self):
        self.data = Table('iris')
        self.louvain = Louvain()

    def test_orange_table(self):
        labels = self.louvain(self.data)
        self.assertEqual(np.ndarray, type(labels))
