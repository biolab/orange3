import unittest

import numpy as np

import Orange
from Orange.evaluation.clustering import Silhouette, \
    AdjustedMutualInfoScore, ClusteringEvaluation, ClusteringResults
from Orange.clustering.kmeans import KMeans

class ClusteringEvaluation_Test(unittest.TestCase):
    def test_init(self):
        res = ClusteringResults(data=Orange.data.Table(np.arange(100).reshape((100, 1))),
                                nmethods=2, nrows=100)
        res.actual[:50] = 0
        res.actual[50:] = 1
        res.predicted = np.vstack((res.actual, res.actual))
        np.testing.assert_almost_equal(AdjustedMutualInfoScore(res), [1.0, 1.0])

    def test_kmeans(self):
        table = Orange.data.Table('iris')
        cr = ClusteringEvaluation(table, learners=[KMeans(n_clusters=2),
                                                   KMeans(n_clusters=3),
                                                   KMeans(n_clusters=5)], k=3)
        np.testing.assert_almost_equal(Silhouette(cr), \
                                       [ 0.68081362,  0.55259194,  0.48851755], decimal=2)
        np.testing.assert_almost_equal(AdjustedMutualInfoScore(cr), \
                                       [0.51936073,  0.74837231,  0.59178896], decimal=2)