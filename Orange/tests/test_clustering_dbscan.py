import unittest

import Orange
from Orange.clustering.dbscan import DBSCAN


class DBSCANTest(unittest.TestCase):


    def test_dbscan_parameters(self):
        table = Orange.data.Table('iris')
        dbscan = DBSCAN(eps=0.1, min_samples=7, metric='euclidean',
                        algorithm='auto', leaf_size=12, p=None,
                        random_state=42)
        c = dbscan(table)

    def test_predict_table(self):
        table = Orange.data.Table('iris')
        dbscan = DBSCAN()
        c = dbscan(table)
        table = table[:20]
        p = c(table)

    def test_predict_numpy(self):
        table = Orange.data.Table('iris')
        dbscan = DBSCAN()
        c = dbscan(table)
        X = table.X[::20]
        p = c(X)

