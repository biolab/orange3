import unittest
import numpy as np
import pickle

import Orange
from Orange.data import Variable
from Orange.preprocess import Continuize
from Orange.projection import PCA, SparsePCA, RandomizedPCA


class TestPCA(unittest.TestCase):
    def test_pca(self):
        data = Orange.data.Table('ionosphere')
        self.__pca_test_helper(data, n_com=3, min_xpl_var=0.5)
        self.__pca_test_helper(data, n_com=10, min_xpl_var=0.7)
        self.__pca_test_helper(data, n_com=32, min_xpl_var=1)

    def __pca_test_helper(self, data, n_com, min_xpl_var):
        pca = PCA(n_components=n_com)
        pca_model = pca(data)
        pca_xpl_var = np.sum(pca_model.explained_variance_ratio_)
        self.assertGreaterEqual(pca_xpl_var + 1e-6, min_xpl_var)
        self.assertEqual(n_com, pca_model.n_components)
        self.assertEqual((n_com, data.X.shape[1]), pca_model.components_.shape)
        proj = np.dot(data.X - pca_model.mean_, pca_model.components_.T)
        np.testing.assert_almost_equal(pca_model(data).X, proj)

    def test_sparse_pca(self):
        data = Orange.data.Table('ionosphere')[:100]
        self.__sparse_pca_test_helper(data, n_com=3, max_err=1500)
        self.__sparse_pca_test_helper(data, n_com=10, max_err=1000)
        self.__sparse_pca_test_helper(data, n_com=32, max_err=500)

    def __sparse_pca_test_helper(self, data, n_com, max_err):
        sparse_pca = SparsePCA(n_components=n_com, ridge_alpha=0.001, random_state=0)
        pca_model = sparse_pca(data)
        self.assertEqual(n_com, pca_model.n_components)
        self.assertEqual((n_com, data.X.shape[1]), pca_model.components_.shape)
        self.assertLessEqual(pca_model.error_[-1], max_err)

    def test_randomized_pca(self):
        data = Orange.data.Table('ionosphere')
        self.__rnd_pca_test_helper(data, n_com=3, min_xpl_var=0.5)
        self.__rnd_pca_test_helper(data, n_com=10, min_xpl_var=0.7)
        self.__rnd_pca_test_helper(data, n_com=32, min_xpl_var=0.98)

    def __rnd_pca_test_helper(self, data, n_com, min_xpl_var):
        rnd_pca = RandomizedPCA(n_components=n_com)
        pca_model = rnd_pca(data)
        pca_xpl_var = np.sum(pca_model.explained_variance_ratio_)
        self.assertGreaterEqual(pca_xpl_var, min_xpl_var)
        self.assertEqual(n_com, pca_model.n_components)
        self.assertEqual((n_com, data.X.shape[1]), pca_model.components_.shape)
        proj = np.dot(data.X - pca_model.mean_, pca_model.components_.T)
        np.testing.assert_almost_equal(pca_model(data).X, proj)

    def test_compute_value(self):
        iris = Orange.data.Table('iris')
        pca = PCA(n_components=2)(iris)
        pca_iris = pca(iris)
        pca_iris2 = Orange.data.Table(pca_iris.domain, iris)
        np.testing.assert_almost_equal(pca_iris.X, pca_iris2.X)
        np.testing.assert_equal(pca_iris.Y, pca_iris2.Y)

        pca_iris3 = pickle.loads(pickle.dumps(pca_iris))
        np.testing.assert_almost_equal(pca_iris.X, pca_iris3.X)
        np.testing.assert_equal(pca_iris.Y, pca_iris3.Y)

    def test_transformed_domain_does_not_pickle_data(self):
        iris = Orange.data.Table('iris')
        pca = PCA(n_components=2)(iris)
        pca_iris = pca(iris)
        pca_iris2 = Orange.data.Table(pca_iris.domain, iris)

        pca_iris2 = pickle.loads(pickle.dumps(pca_iris))
        self.assertIsNone(pca_iris2.domain[0].compute_value.transformed)

    def test_chain(self):
        zoo = Orange.data.Table('zoo')
        zoo_c = Continuize(zoo)
        pca = PCA()(zoo_c)(zoo)
        pca2 = PCA()(zoo_c)(zoo_c)
        pca3 = PCA(preprocessors=[Continuize()])(zoo)(zoo)
        np.testing.assert_almost_equal(pca.X, pca2.X)
        np.testing.assert_almost_equal(pca.X, pca3.X)

