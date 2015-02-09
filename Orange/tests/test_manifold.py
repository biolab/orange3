import unittest
import numpy as np

import Orange
from Orange.projection import MDS, Isomap, LocallyLinearEmbedding
from Orange.distance import Euclidean


class TestManifold(unittest.TestCase):
    def test_manifold(self):
        self._test_mds()
        self._test_isomap()
        self._test_lle()

    def _test_mds(self):
        data = Orange.data.Table('ionosphere')
        self.__mds_test_helper(data, n_com=1)
        self.__mds_test_helper(data, n_com=2)
        self.__mds_test_helper(data, n_com=3)

    def __mds_test_helper(self, data, n_com):
        mds_fit = MDS(n_components=n_com, dissimilarity=Euclidean, random_state=0)
        mds_fit = mds_fit(data)
        mds_dist = MDS(n_components=n_com, dissimilarity='precomputed', random_state=0)
        mds_dist = mds_dist(Euclidean(data))
        eshape = data.X.shape[0], n_com
        self.assertTrue(np.allclose(mds_fit.embedding_, mds_dist.embedding_))
        self.assertEquals(eshape, mds_fit.embedding_.shape)
        self.assertEquals(eshape, mds_dist.embedding_.shape)

    def _test_isomap(self):
        data = Orange.data.Table('ionosphere')
        self.__isomap_test_helper(data, n_com=1)
        self.__isomap_test_helper(data, n_com=2)
        self.__isomap_test_helper(data, n_com=3)

    def __isomap_test_helper(self, data, n_com):
        isomap_fit = Isomap(n_neighbors=5, n_components=n_com)
        isomap_fit = isomap_fit(data)
        eshape = data.X.shape[0], n_com
        self.assertEquals(eshape, isomap_fit.embedding_.shape)

    def _test_lle(self):
        data = Orange.data.Table('ionosphere')
        self.__lle_test_helper(data, n_com=1)
        self.__lle_test_helper(data, n_com=2)
        self.__lle_test_helper(data, n_com=3)

    def __lle_test_helper(self, data, n_com):
        isomap_fit = Isomap(n_neighbors=5, n_components=n_com)
        isomap_fit = isomap_fit(data)
        eshape = data.X.shape[0], n_com
        self.assertEquals(eshape, isomap_fit.embedding_.shape)