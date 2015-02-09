import unittest
import numpy as np

import Orange
from Orange.projection import PCA, SparsePCA, RandomizedPCA


class TestPCA(unittest.TestCase):
    def test_pca(self):
        data = Orange.data.Table('ionosphere')
        self.__pca_test_helper(data, n_com=3, min_xpl_var=0.5)
        self.__pca_test_helper(data, n_com=10, min_xpl_var=0.7)
        self.__pca_test_helper(data, n_com=32, min_xpl_var=1)

    def __pca_test_helper(self, data, n_com, min_xpl_var):
        pca = PCA(n_components=n_com)
        pca = pca(data)
        pca_xpl_var = np.sum(pca.explained_variance_ratio_)
        self.assertGreaterEqual(pca_xpl_var, min_xpl_var)
        self.assertEquals(n_com, pca.n_components)
        self.assertEquals((n_com, data.X.shape[1]), pca.components_.shape)
        proj = np.dot(data.X - pca.mean_, pca.components_.T)
        np.testing.assert_almost_equal(pca.transform(data.X), proj)

    def test_sparse_pca(self):
        data = Orange.data.Table('ionosphere')
        self.__sparse_pca_test_helper(data, n_com=3, max_err=1500)
        self.__sparse_pca_test_helper(data, n_com=10, max_err=1000)
        self.__sparse_pca_test_helper(data, n_com=32, max_err=500)

    def __sparse_pca_test_helper(self, data, n_com, max_err):
        sparse_pca = SparsePCA(n_components=n_com, ridge_alpha=0.001, random_state=0)
        sparse_pca = sparse_pca(data)
        self.assertEquals(n_com, sparse_pca.n_components)
        self.assertEquals((n_com, data.X.shape[1]), sparse_pca.components_.shape)
        self.assertLessEqual(sparse_pca.error_[-1], max_err)

    def test_randomized_pca(self):
        data = Orange.data.Table('ionosphere')
        self.__rnd_pca_test_helper(data, n_com=3, min_xpl_var=0.5)
        self.__rnd_pca_test_helper(data, n_com=10, min_xpl_var=0.7)
        self.__rnd_pca_test_helper(data, n_com=32, min_xpl_var=0.98)

    def __rnd_pca_test_helper(self, data, n_com, min_xpl_var):
        rnd_pca = RandomizedPCA(n_components=n_com)
        rnd_pca = rnd_pca(data)
        pca_xpl_var = np.sum(rnd_pca.explained_variance_ratio_)
        self.assertGreaterEqual(pca_xpl_var, min_xpl_var)
        self.assertEquals(n_com, rnd_pca.n_components)
        self.assertEquals((n_com, data.X.shape[1]), rnd_pca.components_.shape)
        proj = np.dot(data.X - rnd_pca.mean_, rnd_pca.components_.T)
        np.testing.assert_almost_equal(rnd_pca.transform(data.X), proj)
