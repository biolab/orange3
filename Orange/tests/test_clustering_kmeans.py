# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring

import unittest
import warnings

import numpy as np
from scipy.sparse import csc_matrix, csr_matrix

import Orange
from Orange.clustering.kmeans import KMeans, KMeansModel
from Orange.data import Table, Domain, ContinuousVariable
from Orange.data.table import DomainTransformationError
from Orange.tests.test_dasktable import with_dasktable


class TestKMeans(unittest.TestCase):
    def setUp(self):
        self.kmeans = KMeans(n_clusters=2)
        self.iris = Orange.data.Table('iris')

    @with_dasktable
    def test_kmeans(self, prepare_table):
        iris = prepare_table(self.iris)
        c = self.kmeans(iris)
        # First 20 iris belong to one cluster
        self.assertEqual(type(iris.X), type(c))
        self.assertEqual(len(iris), len(c))
        self.assertEqual(1, len(np.unique(np.asarray(c[:20]))))

    @with_dasktable
    def test_kmeans_parameters(self, prepare_table):
        kmeans = KMeans(n_clusters=10, max_iter=10, random_state=42, tol=0.001,
                        init='random')
        iris = prepare_table(self.iris)
        c = kmeans(iris)
        self.assertEqual(type(iris.X), type(c))
        self.assertEqual(len(iris), len(c))

    @with_dasktable
    def test_predict_table(self, prepare_table):
        iris = prepare_table(self.iris)
        c = self.kmeans(iris)
        self.assertEqual(type(iris.X), type(c))
        self.assertEqual(len(iris), len(c))

    @with_dasktable
    def test_predict_numpy(self, prepare_table):
        iris = prepare_table(self.iris)
        c = self.kmeans.fit(iris.X)
        self.assertEqual(KMeansModel, type(c))
        self.assertEqual(type(iris.X), type(c.labels))
        self.assertEqual(len(iris), len(c.labels))

    def test_predict_sparse_csc(self):
        with self.iris.unlocked():
            self.iris.X = csc_matrix(self.iris.X[::20])
        c = self.kmeans(self.iris)
        self.assertEqual(np.ndarray, type(c))
        self.assertEqual(len(self.iris), len(c))

    def test_predict_spares_csr(self):
        with self.iris.unlocked():
            self.iris.X = csr_matrix(self.iris.X[::20])
        c = self.kmeans(self.iris)
        self.assertEqual(np.ndarray, type(c))
        self.assertEqual(len(self.iris), len(c))

    @with_dasktable
    def test_model(self, prepare_table):
        iris = prepare_table(self.iris)
        c = self.kmeans.get_model(iris)
        self.assertEqual(KMeansModel, type(c))
        self.assertEqual(len(iris), len(c.labels))

        c1 = c(iris)
        # prediction of the model must be same since data are same
        np.testing.assert_array_almost_equal(c.labels, c1)

    @with_dasktable
    def test_model_np(self, prepare_table):
        """
        Test with numpy array as an input in model.
        """
        iris = prepare_table(self.iris)
        c = self.kmeans.get_model(iris)
        c1 = c(iris.X)
        # prediction of the model must be same since data are same
        np.testing.assert_array_almost_equal(c.labels, c1)

    def test_model_sparse_csc(self):
        """
        Test with sparse array as an input in model.
        """
        c = self.kmeans.get_model(self.iris)
        c1 = c(csc_matrix(self.iris.X))
        # prediction of the model must be same since data are same
        np.testing.assert_array_almost_equal(c.labels, c1)

    def test_model_sparse_csr(self):
        """
        Test with sparse array as an input in model.
        """
        c = self.kmeans.get_model(self.iris)
        c1 = c(csr_matrix(self.iris.X))
        # prediction of the model must be same since data are same
        np.testing.assert_array_almost_equal(c.labels, c1)

    @with_dasktable
    def test_model_instance(self, prepare_table):
        """
        Test with instance as an input in model.
        """
        iris = prepare_table(self.iris)
        c = self.kmeans.get_model(iris)
        c1 = c(iris[0])
        # prediction of the model must be same since data are same
        self.assertEqual(c1, c.labels[0])

    def test_model_list(self):
        """
        Test with list as an input in model.
        """
        c = self.kmeans.get_model(self.iris)
        c1 = c(np.asarray(self.iris.X).tolist())
        # prediction of the model must be same since data are same
        np.testing.assert_array_almost_equal(c.labels, c1)

        # example with a list of only one data item
        c1 = c(np.asarray(self.iris.X).tolist()[0])
        # prediction of the model must be same since data are same
        np.testing.assert_array_almost_equal(c.labels[0], c1)

    @with_dasktable
    def test_model_bad_datatype(self, prepare_table):
        """
        Check model with data-type that is not supported.
        """
        iris = prepare_table(self.iris)
        c = self.kmeans.get_model(iris)
        self.assertRaises(TypeError, c, 10)

    def test_model_data_table_domain(self):
        """
        Check model with data-type that is not supported.
        """
        # ok domain
        data = Table(Domain(
            list(self.iris.domain.attributes) + [ContinuousVariable("a")]),
                     np.concatenate((self.iris.X, np.ones((len(self.iris), 1))), axis=1))
        c = self.kmeans.get_model(self.iris)
        res = c(data)
        np.testing.assert_array_almost_equal(c.labels, res)

        # totally different domain - should fail
        self.assertRaises(DomainTransformationError, c, Table("housing"))

    def test_deprecated_silhouette(self):
        with warnings.catch_warnings(record=True) as w:
            KMeans(compute_silhouette_score=True)

            assert len(w) == 1
            assert issubclass(w[-1].category, DeprecationWarning)

        with warnings.catch_warnings(record=True) as w:
            KMeans(compute_silhouette_score=False)

            assert len(w) == 1
            assert issubclass(w[-1].category, DeprecationWarning)
