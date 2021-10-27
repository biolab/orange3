# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring
import pickle
import platform
import unittest

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

from Orange.data import Table
from Orange.distance import Euclidean
from Orange.projection import (MDS, Isomap, LocallyLinearEmbedding,
                               SpectralEmbedding, TSNE)
from Orange.projection.manifold import torgerson
from Orange.tests import test_filename


np.random.seed(42)


class TestManifold(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ionosphere = Table(test_filename('datasets/ionosphere.tab'))
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


class TestTSNE(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.iris = Table('iris')

    def test_fit(self):
        n_components = 2
        tsne = TSNE(n_components=n_components)
        model = tsne(self.iris)

        # The embedding should have the correct number of dimensions
        self.assertEqual(model.embedding.X.shape, (self.iris.X.shape[0], n_components))

        # The embedding should not contain NaNs
        self.assertFalse(np.any(np.isnan(model.embedding.X)))

        # The embeddings in the table should match the embedding object
        np.testing.assert_equal(model.embedding.X, model.embedding_)

    def test_transform(self):
        # Set perplexity to avoid warnings
        tsne = TSNE(perplexity=10)
        model = tsne(self.iris[::2])
        new_embedding = model(self.iris[1::2])

        # The new embedding should not contain NaNs
        self.assertFalse(np.any(np.isnan(new_embedding.X)))

    def test_multiscale(self):
        tsne = TSNE(perplexity=(10, 10), multiscale=True)
        model = tsne(self.iris[::2])
        embedding = model(self.iris[1::2])
        self.assertFalse(np.any(np.isnan(embedding.X)))

    def test_continue_optimization(self):
        tsne = TSNE(n_iter=100)
        model = tsne(self.iris)
        new_model = model.optimize(100, inplace=False)

        # If we don't do things inplace, then the instances should be different
        self.assertIsNot(model, new_model)
        self.assertIsNot(model.embedding, new_model.embedding)
        self.assertIsNot(model.embedding_, new_model.embedding_)

        self.assertFalse(np.allclose(model.embedding.X, new_model.embedding.X),
                         'Embedding should change after further optimization.')

        # The embeddings in the table should match the embedding object
        np.testing.assert_equal(new_model.embedding.X, new_model.embedding_)

    def test_continue_optimization_inplace(self):
        tsne = TSNE(n_iter=100)
        model = tsne(self.iris)
        new_model = model.optimize(100, inplace=True)

        # If we don't do things inplace, then the instances should be the same
        self.assertIs(model, new_model)
        self.assertIs(model.embedding, new_model.embedding)
        self.assertIs(model.embedding_, new_model.embedding_)

        # The embeddings in the table should match the embedding object
        np.testing.assert_equal(new_model.embedding.X, new_model.embedding_)

    def test_bh_correctness(self):
        knn = KNeighborsClassifier(n_neighbors=5)

        # Set iterations to 0 so we check that the initialization is fairly random
        tsne = TSNE(early_exaggeration_iter=0, n_iter=0, perplexity=30,
                    negative_gradient_method='bh', initialization='random',
                    random_state=0)
        model = tsne(self.iris)

        # Evaluate KNN on the random initialization
        knn.fit(model.embedding_, self.iris.Y)
        predicted = knn.predict(model.embedding_)
        self.assertTrue(accuracy_score(predicted, self.iris.Y) < 0.6)

        # 100 iterations should be enough for iris
        model.optimize(n_iter=100, inplace=True)

        # Evaluate KNN on the tSNE embedding
        knn.fit(model.embedding_, self.iris.Y)
        predicted = knn.predict(model.embedding_)
        self.assertTrue(accuracy_score(predicted, self.iris.Y) > 0.95)

    def test_fft_correctness(self):
        knn = KNeighborsClassifier(n_neighbors=5)

        # Set iterations to 0 so we check that the initialization is fairly random
        tsne = TSNE(early_exaggeration_iter=0, n_iter=0, perplexity=30,
                    negative_gradient_method='fft', initialization='random',
                    random_state=0)
        model = tsne(self.iris)

        # Evaluate KNN on the random initialization
        knn.fit(model.embedding_, self.iris.Y)
        predicted = knn.predict(model.embedding_)
        self.assertTrue(accuracy_score(predicted, self.iris.Y) < 0.6)

        # 100 iterations should be enough for iris
        model.optimize(n_iter=100, inplace=True)

        # Evaluate KNN on the tSNE embedding
        knn.fit(model.embedding_, self.iris.Y)
        predicted = knn.predict(model.embedding_)
        self.assertTrue(accuracy_score(predicted, self.iris.Y) > 0.95)

    @unittest.skipIf(platform.system() == "Windows", "Files locked on Windows")
    def test_pickle(self):
        for neighbors in ("exact", "approx"):
            tsne = TSNE(early_exaggeration_iter=0, n_iter=10, perplexity=30,
                        neighbors=neighbors, random_state=0)
            model = tsne(self.iris[::2])

            loaded_model = pickle.loads(pickle.dumps(model))

            new_embedding = loaded_model(self.iris[1::2]).X

            knn = KNeighborsClassifier(n_neighbors=5)
            knn.fit(new_embedding, self.iris[1::2].Y)
            predicted = knn.predict(new_embedding)
            self.assertTrue(
                accuracy_score(predicted, self.iris[1::2].Y) > 0.95,
                msg=f"Pickling failed with `neighbors={neighbors}`",
            )
