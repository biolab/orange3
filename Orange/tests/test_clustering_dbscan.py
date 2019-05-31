# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring

import unittest

import Orange
from Orange.clustering.dbscan import DBSCAN


class TestDBSCAN(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.iris = Orange.data.Table('iris')

    def test_dbscan_parameters(self):
        dbscan = DBSCAN(eps=0.1, min_samples=7, metric='euclidean',
                        algorithm='auto', leaf_size=12, p=None)
        c = dbscan(self.iris)

    def test_predict_table(self):
        dbscan = DBSCAN()
        c = dbscan(self.iris)
        table = self.iris[:20]
        p = c(table)

    def test_predict_numpy(self):
        dbscan = DBSCAN()
        c = dbscan(self.iris)
        X = self.iris.X[::20]
        p = c(X)

    def test_values(self):
        dbscan = DBSCAN(eps=1)  # it clusters data in two classes
        c = dbscan(self.iris)
        table = self.iris
        p = c(table)

        self.assertEqual(2, len(p.domain[0].values))
        self.assertSetEqual({"0", "1"}, set(p.domain[0].values))

        table.X[0] = [100, 100, 100, 100]  # we add a big outlier

        p = c(table)

        self.assertEqual(3, len(p.domain[0].values))
        self.assertSetEqual({"-1", "0", "1"}, set(p.domain[0].values))
