# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring

import unittest
import numpy as np

from Orange.projection import (MDS, Isomap, LocallyLinearEmbedding,
                               SpectralEmbedding, TSNE)
from Orange.projection.manifold import torgerson
from Orange.distance import Euclidean
from Orange.data import Table


class TestManifold(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ionosphere = Table('ionosphere')
        cls.iris = Table('iris')

    def test_mds(self):
        data = self.ionosphere[:50]
        for i in range(1, 4):
            self.__mds_test_helper(data, n_com=i)

    def __mds_test_helper(self, data, n_com):
        mds_fit = MDS(
            n_components=n_com, dissimilarity=Euclidean, random_state=0)
        mds_fit = mds_fit(data)

        mds_odist = MDS(
            n_components=n_com, dissimilarity='precomputed', random_state=0)
        mds_odist = mds_odist(Euclidean(data))

        mds_sdist = MDS(
            n_components=n_com, dissimilarity='euclidean', random_state=0)
        mds_sdist = mds_sdist(data)

        eshape = data.X.shape[0], n_com
        self.assertTrue(np.allclose(mds_fit.embedding_, mds_odist.embedding_))
        self.assertTrue(np.allclose(mds_fit.embedding_, mds_sdist.embedding_))
        self.assertEqual(eshape, mds_fit.embedding_.shape)
        self.assertEqual(eshape, mds_odist.embedding_.shape)
        self.assertEqual(eshape, mds_sdist.embedding_.shape)

    def test_mds_pca_init(self):
        result = np.array([-2.6928912, 0.32603512])

        projector = MDS(
            n_components=2, dissimilarity=Euclidean, init_type='PCA',
            n_init=1)
        X = projector(self.iris).embedding_
        np.testing.assert_array_almost_equal(X[0], result)

        projector = MDS(
            n_components=2, dissimilarity='precomputed', init_type='PCA',
            n_init=1)
        X = projector(Euclidean(self.iris)).embedding_
        np.testing.assert_array_almost_equal(X[0], result)

        projector = MDS(
            n_components=2, dissimilarity='euclidean', init_type='PCA',
            n_init=1)
        X = projector(self.iris).embedding_
        np.testing.assert_array_almost_equal(X[0], result)

        projector = MDS(
            n_components=6, dissimilarity='euclidean', init_type='PCA',
            n_init=1)
        X = projector(self.iris[:5]).embedding_
        result = np.array([-0.31871, -0.064644, 0.015653, -1.5e-08, -4.3e-11, 0])
        np.testing.assert_array_almost_equal(np.abs(X[0]), np.abs(result))

    def test_isomap(self):
        for i in range(1, 4):
            self.__isomap_test_helper(self.ionosphere, n_com=i)

    def __isomap_test_helper(self, data, n_com):
        isomap_fit = Isomap(n_neighbors=5, n_components=n_com)
        isomap_fit = isomap_fit(data)
        eshape = data.X.shape[0], n_com
        self.assertEqual(eshape, isomap_fit.embedding_.shape)

    def test_lle(self):
        for i in range(1, 4):
            self.__lle_test_helper(self.ionosphere, n_com=i)

    def __lle_test_helper(self, data, n_com):
        lle = LocallyLinearEmbedding(n_neighbors=5, n_components=n_com)
        lle = lle(data)

        ltsa = LocallyLinearEmbedding(n_neighbors=5, n_components=n_com,
                                      method="ltsa",
                                      eigen_solver="dense")
        ltsa = ltsa(data)

        hessian = LocallyLinearEmbedding(n_neighbors=15, n_components=n_com,
                                         method="hessian",
                                         eigen_solver="dense")
        hessian = hessian(data)

        modified = LocallyLinearEmbedding(n_neighbors=5, n_components=n_com,
                                          method="modified",
                                          eigen_solver="dense")
        modified = modified(data)

        self.assertEqual((data.X.shape[0], n_com), lle.embedding_.shape)
        self.assertEqual((data.X.shape[0], n_com), ltsa.embedding_.shape)
        self.assertEqual((data.X.shape[0], n_com), hessian.embedding_.shape)
        self.assertEqual((data.X.shape[0], n_com), modified.embedding_.shape)

    def test_se(self):
        for i in range(1, 4):
            self.__se_test_helper(self.ionosphere, n_com=i)

    def __se_test_helper(self, data, n_com):
        se = SpectralEmbedding(n_components=n_com, n_neighbors=5)
        se = se(data)
        self.assertEqual((data.X.shape[0], n_com), se.embedding_.shape)

    def test_tsne(self):
        data = self.ionosphere[:50]
        for i in range(1, 4):
            self.__tsne_test_helper(data, n_com=i)

    def __tsne_test_helper(self, data, n_com):
        tsne_def = TSNE(n_components=n_com, metric='euclidean')
        tsne_def = tsne_def(data)

        tsne_euc = TSNE(n_components=n_com, metric=Euclidean)
        tsne_euc = tsne_euc(data)

        tsne_pre = TSNE(n_components=n_com, metric='precomputed')
        tsne_pre = tsne_pre(Euclidean(data))

        self.assertEqual((data.X.shape[0], n_com), tsne_def.embedding_.shape)
        self.assertEqual((data.X.shape[0], n_com), tsne_euc.embedding_.shape)
        self.assertEqual((data.X.shape[0], n_com), tsne_pre.embedding_.shape)

    def test_torgerson(self):
        data = self.ionosphere[::5]
        dis = Euclidean(data)

        e1 = torgerson(dis, eigen_solver="auto")
        e2 = torgerson(dis, eigen_solver="lapack")
        e3 = torgerson(dis, eigen_solver="arpack")

        np.testing.assert_almost_equal(np.abs(e1), np.abs(e2))
        np.testing.assert_almost_equal(np.abs(e2), np.abs(e3))

        with self.assertRaises(ValueError):
            torgerson(dis, eigen_solver="madness")
