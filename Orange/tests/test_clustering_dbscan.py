# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring

import unittest

import numpy as np

from Orange.data import Table
from Orange.clustering.dbscan import DBSCAN


class TestDBSCAN(unittest.TestCase):
    def setUp(self):
        self.iris = Table('iris')
        self.dbscan = DBSCAN()

    def test_dbscan_parameters(self):
        dbscan = DBSCAN(eps=0.1, min_samples=7, metric='euclidean',
                        algorithm='auto', leaf_size=12, p=None)
        dbscan(self.iris)

    def test_predict_table(self):
        pred = self.dbscan(self.iris)
        self.assertEqual(np.ndarray, type(pred))

    def test_predict_numpy(self):
        model = self.dbscan.fit(self.iris.X)
        self.assertEqual(np.ndarray, type(model.labels))
