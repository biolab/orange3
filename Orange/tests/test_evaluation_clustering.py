# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring

import unittest

import numpy as np

import Orange
from Orange.evaluation.clustering import Silhouette, \
    AdjustedMutualInfoScore, ClusteringEvaluation, ClusteringResults
from Orange.clustering.kmeans import KMeans, KMeansModel


class TestClusteringResults(unittest.TestCase):
    @staticmethod
    def test_init():
        data = Orange.data.Table.from_numpy(
            None, np.arange(100).reshape((100, 1)))
        res = ClusteringResults(data=data, nmethods=2, nrows=100)
        res.actual[:50] = 0
        res.actual[50:] = 1
        res.predicted = np.vstack((res.actual, res.actual))
        expected = [1.0, 1.0]
        np.testing.assert_almost_equal(AdjustedMutualInfoScore(res), expected)


class TestClusteringEvaluation(unittest.TestCase):
    def test_init(self):
        res = ClusteringEvaluation(k=42)
        self.assertEqual(res.k, 42)

    def test_kmeans(self):
        table = Orange.data.Table('iris')
        cr = ClusteringEvaluation(k=3)(table, learners=[KMeans(n_clusters=2),
                                                        KMeans(n_clusters=3),
                                                        KMeans(n_clusters=5)])
        expected = [0.68081362, 0.55259194, 0.48851755]
        np.testing.assert_almost_equal(Silhouette(cr), expected, decimal=2)
        expected = [0.65383807, 0.75511917, 0.68721092]
        np.testing.assert_almost_equal(AdjustedMutualInfoScore(cr),
                                       expected, decimal=2)
        self.assertIsNone(cr.models)

        cr = ClusteringEvaluation(k=3, store_models=True)(
            table, learners=[KMeans(n_clusters=2)])
        self.assertEqual(cr.models.shape, (3, 1))
        self.assertTrue(all(isinstance(m, KMeansModel)
                            for m in cr.models.flatten()))
