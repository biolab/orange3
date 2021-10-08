# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring

import unittest

import numpy as np
from scipy.sparse import csc_matrix, csr_matrix

from Orange.clustering.clustering import ClusteringModel
from Orange.data import Table
from Orange.clustering.dbscan import DBSCAN


class TestDBSCAN(unittest.TestCase):
    def setUp(self):
        self.iris = Table('iris')
        self.dbscan = DBSCAN()

    def test_dbscan(self):
        c = self.dbscan(self.iris)
        # First 20 iris belong to one cluster
        self.assertEqual(np.ndarray, type(c))
        self.assertEqual(len(self.iris), len(c))
        self.assertEqual(1, len(set(c[:20].ravel())))

    def test_dbscan_parameters(self):
        dbscan = DBSCAN(eps=0.1, min_samples=7, metric='euclidean',
                        algorithm='auto', leaf_size=12, p=None)
        c = dbscan(self.iris)
        self.assertEqual(np.ndarray, type(c))
        self.assertEqual(len(self.iris), len(c))

    def test_predict_table(self):
        pred = self.dbscan(self.iris)
        self.assertEqual(np.ndarray, type(pred))
        self.assertEqual(len(self.iris), len(pred))

    def test_predict_numpy(self):
        model = self.dbscan.fit(self.iris.X)
        self.assertEqual(ClusteringModel, type(model))
        self.assertEqual(np.ndarray, type(model.labels))
        self.assertEqual(len(self.iris), len(model.labels))

    def test_predict_sparse_csc(self):
        with self.iris.unlocked():
            self.iris.X = csc_matrix(self.iris.X[::20])
        c = self.dbscan(self.iris)
        self.assertEqual(np.ndarray, type(c))
        self.assertEqual(len(self.iris), len(c))

    def test_predict_spares_csr(self):
        with self.iris.unlocked():
            self.iris.X = csr_matrix(self.iris.X[::20])
        c = self.dbscan(self.iris)
        self.assertEqual(np.ndarray, type(c))
        self.assertEqual(len(self.iris), len(c))

    def test_model(self):
        c = self.dbscan.get_model(self.iris)
        self.assertEqual(ClusteringModel, type(c))
        self.assertEqual(len(self.iris), len(c.labels))

        self.assertRaises(NotImplementedError, c, self.iris)

    def test_model_np(self):
        """
        Test with numpy array as an input in model.
        """
        c = self.dbscan.get_model(self.iris)
        self.assertRaises(NotImplementedError, c, self.iris.X)

    def test_model_sparse(self):
        """
        Test with sparse array as an input in model.
        """
        c = self.dbscan.get_model(self.iris)
        self.assertRaises(NotImplementedError, c, csr_matrix(self.iris.X))

    def test_model_instance(self):
        """
        Test with instance as an input in model.
        """
        c = self.dbscan.get_model(self.iris)
        self.assertRaises(NotImplementedError, c, self.iris[0])

    def test_model_list(self):
        """
        Test with list as an input in model.
        """
        c = self.dbscan.get_model(self.iris)
        self.assertRaises(NotImplementedError, c, self.iris.X.tolist())

    def test_model_bad_datatype(self):
        """
        Check model with data-type that is not supported.
        """
        c = self.dbscan.get_model(self.iris)
        self.assertRaises(TypeError, c, 10)
