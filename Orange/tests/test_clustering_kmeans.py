import unittest

import Orange
from Orange.clustering.kmeans import KMeans


class KMeansTest(unittest.TestCase):

    def test_kmeans(self):
        table = Orange.data.Table('iris')
        kmeans = KMeans(n_clusters=2)
        c = kmeans(table)
        X = table.X[:20]
        p = c(X)
        # First 20 iris belong to one cluster
        assert len(set(p.ravel())) == 1

    def test_kmeans_parameters(self):
        table = Orange.data.Table('iris')
        kmeans = KMeans(n_clusters=20,
                        max_iter=10,
                        random_state=42,
                        tol=0.001,
                        init='random')
        c = kmeans(table)

    def test_predict_single_instance(self):
        table = Orange.data.Table('iris')
        kmeans = KMeans()
        c = kmeans(table)
        inst = table[0]
        p = c(inst)

    def test_predict_table(self):
        table = Orange.data.Table('iris')
        kmeans = KMeans()
        c = kmeans(table)
        table = table[:20]
        p = c(table)

    def test_predict_numpy(self):
        table = Orange.data.Table('iris')
        kmeans = KMeans()
        c = kmeans(table)
        X = table.X[::20]
        p = c(X)

