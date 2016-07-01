# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring

import unittest

import Orange
from Orange.clustering.kmeans import KMeans


class TestKMeans(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.iris = Orange.data.Table('iris')

    def test_kmeans(self):
        kmeans = KMeans(n_clusters=2)
        c = kmeans(self.iris)
        X = self.iris.X[:20]
        p = c(X)
        # First 20 iris belong to one cluster
        assert len(set(p.ravel())) == 1

    def test_kmeans_parameters(self):
        kmeans = KMeans(n_clusters=10,
                        max_iter=10,
                        random_state=42,
                        tol=0.001,
                        init='random',
                        compute_silhouette_score=True)
        c = kmeans(self.iris)

    def test_predict_single_instance(self):
        kmeans = KMeans()
        c = kmeans(self.iris)
        inst = self.iris[0]
        p = c(inst)

    def test_predict_table(self):
        kmeans = KMeans()
        c = kmeans(self.iris)
        table = self.iris[:20]
        p = c(table)

    def test_predict_numpy(self):
        kmeans = KMeans()
        c = kmeans(self.iris)
        X = self.iris.X[::20]
        p = c(X)

