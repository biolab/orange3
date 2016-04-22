# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring

import unittest
import numpy as np

from Orange.projection import MDS, Isomap
from Orange.distance import Euclidean
from Orange.data import Table


class TestManifold(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ionosphere = Table('ionosphere')

    def test_mds(self):
        data = self.ionosphere[:50]
        for i in range(1, 4):
            self.__mds_test_helper(data, n_com=i)

    def __mds_test_helper(self, data, n_com):
        mds_fit = MDS(n_components=n_com, dissimilarity=Euclidean, random_state=0)
        mds_fit = mds_fit(data)

        mds_odist = MDS(n_components=n_com, dissimilarity='precomputed', random_state=0)
        mds_odist = mds_odist(Euclidean(data))

        mds_sdist = MDS(n_components=n_com, dissimilarity='euclidean', random_state=0)
        mds_sdist = mds_sdist(data)

        eshape = data.X.shape[0], n_com
        self.assertTrue(np.allclose(mds_fit.embedding_, mds_odist.embedding_))
        self.assertTrue(np.allclose(mds_fit.embedding_, mds_sdist.embedding_))
        self.assertEqual(eshape, mds_fit.embedding_.shape)
        self.assertEqual(eshape, mds_odist.embedding_.shape)
        self.assertEqual(eshape, mds_sdist.embedding_.shape)

    def test_isomap(self):
        for i in range(1, 4):
            self.__isomap_test_helper(self.ionosphere, n_com=i)

    def __isomap_test_helper(self, data, n_com):
        isomap_fit = Isomap(n_neighbors=5, n_components=n_com)
        isomap_fit = isomap_fit(data)
        eshape = data.X.shape[0], n_com
        self.assertEqual(eshape, isomap_fit.embedding_.shape)
