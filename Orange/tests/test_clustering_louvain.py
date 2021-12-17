# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring

import unittest

import numpy as np
import networkx
from scipy.sparse import csc_matrix, csr_matrix

from Orange.clustering.clustering import ClusteringModel
from Orange.clustering.louvain import matrix_to_knn_graph
from Orange.data import Table
from Orange.clustering.louvain import Louvain


class TestLouvain(unittest.TestCase):
    def setUp(self):
        self.iris = Table('iris')
        self.louvain = Louvain()

    def test_louvain(self):
        c = self.louvain(self.iris)
        # First 20 iris belong to one cluster
        self.assertEqual(np.ndarray, type(c))
        self.assertEqual(len(self.iris), len(c))
        self.assertEqual(1, len(set(c[:20].ravel())))

    def test_louvain_parameters(self):
        louvain = Louvain(
            k_neighbors=3, resolution=1.2, random_state=42, metric="l2")
        c = louvain(self.iris)
        self.assertEqual(np.ndarray, type(c))
        self.assertEqual(len(self.iris), len(c))

    def test_predict_table(self):
        c = self.louvain(self.iris)
        self.assertEqual(np.ndarray, type(c))
        self.assertEqual(len(self.iris), len(c))

    def test_predict_numpy(self):
        c = self.louvain.fit(self.iris.X)
        self.assertEqual(ClusteringModel, type(c))
        self.assertEqual(np.ndarray, type(c.labels))
        self.assertEqual(len(self.iris), len(c.labels))

    def test_predict_sparse_csc(self):
        with self.iris.unlocked():
            self.iris.X = csc_matrix(self.iris.X[::5])
        c = self.louvain(self.iris)
        self.assertEqual(np.ndarray, type(c))
        self.assertEqual(len(self.iris), len(c))

    def test_predict_sparse_csr(self):
        with self.iris.unlocked():
            self.iris.X = csr_matrix(self.iris.X[::5])
        c = self.louvain(self.iris)
        self.assertEqual(np.ndarray, type(c))
        self.assertEqual(len(self.iris), len(c))

    def test_model(self):
        c = self.louvain.get_model(self.iris)
        self.assertEqual(ClusteringModel, type(c))
        self.assertEqual(len(self.iris), len(c.labels))

        self.assertRaises(NotImplementedError, c, self.iris)

    def test_model_np(self):
        """
        Test with numpy array as an input in model.
        """
        c = self.louvain.get_model(self.iris)
        self.assertRaises(NotImplementedError, c, self.iris.X)

    def test_model_sparse(self):
        """
        Test with sparse array as an input in model.
        """
        c = self.louvain.get_model(self.iris)
        self.assertRaises(NotImplementedError, c, csr_matrix(self.iris.X))

    def test_model_instance(self):
        """
        Test with instance as an input in model.
        """
        c = self.louvain.get_model(self.iris)
        self.assertRaises(NotImplementedError, c, self.iris[0])

    def test_model_list(self):
        """
        Test with list as an input in model.
        """
        c = self.louvain.get_model(self.iris)
        self.assertRaises(NotImplementedError, c, self.iris.X.tolist())

    def test_graph(self):
        """
        Louvain accepts graphs too.
        :return:
        """
        graph = matrix_to_knn_graph(self.iris.X, 30, "l2")
        self.assertIsNotNone(graph)
        self.assertEqual(networkx.Graph, type(graph), 1)

        # basic clustering - get clusters
        c = self.louvain(graph)
        # First 20 iris belong to one cluster
        self.assertEqual(np.ndarray, type(c))
        self.assertEqual(len(self.iris), len(c))
        self.assertEqual(1, len(set(c[:20].ravel())))

        # clustering - get model
        c = self.louvain.get_model(graph)
        # First 20 iris belong to one cluster
        self.assertEqual(ClusteringModel, type(c))
        self.assertEqual(len(self.iris), len(c.labels))

    def test_model_bad_datatype(self):
        """
        Check model with data-type that is not supported.
        """
        c = self.louvain.get_model(self.iris)
        self.assertRaises(TypeError, c, 10)
