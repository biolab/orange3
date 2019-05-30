# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring

import unittest

import numpy as np
from scipy.sparse import csc_matrix

import Orange
from Orange.clustering.kmeans import KMeans


class TestKMeans(unittest.TestCase):
    def setUp(self):
        self.kmeans = KMeans(n_clusters=2)
        self.iris = Orange.data.Table('iris')

    def test_kmeans(self):
        c = self.kmeans(self.iris)
        # First 20 iris belong to one cluster
        self.assertEqual(1, len(set(c[:20].ravel())))

    def test_kmeans_parameters(self):
        kmeans = KMeans(n_clusters=10, max_iter=10, random_state=42, tol=0.001,
                        init='random')
        kmeans(self.iris)

    def test_predict_table(self):
        kmeans = KMeans()
        c = kmeans(self.iris)
        self.assertEqual(np.ndarray, type(c))

    def test_predict_numpy(self):
        kmeans = KMeans()
        c = kmeans.fit(self.iris.X)
        self.assertEqual(np.ndarray, type(c.labels))

    def test_predict_sparse(self):
        kmeans = KMeans()
        self.iris.X = csc_matrix(self.iris.X[::20])
        c = kmeans(self.iris)
        self.assertEqual(np.ndarray, type(c))
