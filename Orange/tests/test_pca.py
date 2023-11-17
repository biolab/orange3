# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring
import pickle
import unittest
from unittest.mock import MagicMock

import numpy as np
from sklearn.utils import check_random_state

from Orange.data import Table, Domain
from Orange.preprocess import Continuize, Normalize
from Orange.projection import pca, PCA, SparsePCA, IncrementalPCA, TruncatedSVD
from Orange.tests import test_filename


class TestPCA(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ionosphere = Table(test_filename('datasets/ionosphere.tab'))
        cls.iris = Table('iris')
        cls.zoo = Table('zoo')

    def test_pca(self):
        data = self.ionosphere
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
        data = self.ionosphere[:100]
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
        data = self.ionosphere
        self.__rnd_pca_test_helper(data, n_com=3, min_xpl_var=0.5)
        self.__rnd_pca_test_helper(data, n_com=10, min_xpl_var=0.7)
        self.__rnd_pca_test_helper(data, n_com=32, min_xpl_var=0.98)

    def __rnd_pca_test_helper(self, data, n_com, min_xpl_var):
        rnd_pca = PCA(n_components=n_com, svd_solver='randomized')
        pca_model = rnd_pca(data)
        pca_xpl_var = np.sum(pca_model.explained_variance_ratio_)
        self.assertGreaterEqual(pca_xpl_var, min_xpl_var)
        self.assertEqual(n_com, pca_model.n_components)
        self.assertEqual((n_com, data.X.shape[1]), pca_model.components_.shape)
        proj = np.dot(data.X - pca_model.mean_, pca_model.components_.T)
        np.testing.assert_almost_equal(pca_model(data).X, proj)

    def test_improved_randomized_pca_properly_called(self):
        # It doesn't matter what we put into the matrix
        x_ = np.random.normal(0, 1, (100, 20))
        x = Table.from_numpy(Domain.from_numpy(x_), x_)

        pca.randomized_pca = MagicMock(wraps=pca.randomized_pca)
        PCA(10, svd_solver="randomized", random_state=42)(x)
        pca.randomized_pca.assert_called_once()

        pca.randomized_pca.reset_mock()
        PCA(10, svd_solver="arpack", random_state=42)(x)
        pca.randomized_pca.assert_not_called()

    def test_improved_randomized_pca_dense_data(self):
        """Randomized PCA should work well on dense data."""
        random_state = check_random_state(42)

        # Let's take a tall, skinny matrix
        x_ = random_state.normal(0, 1, (100, 20))
        x = Table.from_numpy(Domain.from_numpy(x_), x_)

        pca = PCA(10, svd_solver="full", random_state=random_state)(x)
        rpca = PCA(10, svd_solver="randomized", random_state=random_state)(x)

        np.testing.assert_almost_equal(
            pca.components_, rpca.components_, decimal=8
        )
        np.testing.assert_almost_equal(
            pca.explained_variance_, rpca.explained_variance_, decimal=8
        )
        np.testing.assert_almost_equal(
            pca.singular_values_, rpca.singular_values_, decimal=8
        )

        # And take a short, fat matrix
        x_ = random_state.normal(0, 1, (20, 100))
        x = Table.from_numpy(Domain.from_numpy(x_), x_)

        pca = PCA(10, svd_solver="full", random_state=random_state)(x)
        rpca = PCA(10, svd_solver="randomized", random_state=random_state)(x)

        np.testing.assert_almost_equal(
            pca.components_, rpca.components_, decimal=8
        )
        np.testing.assert_almost_equal(
            pca.explained_variance_, rpca.explained_variance_, decimal=8
        )
        np.testing.assert_almost_equal(
            pca.singular_values_, rpca.singular_values_, decimal=8
        )

    def test_improved_randomized_pca_sparse_data(self):
        """Randomized PCA should work well on dense data."""
        random_state = check_random_state(42)

        # Let's take a tall, skinny matrix
        x_ = random_state.negative_binomial(1, 0.5, (100, 20))
        x = Table.from_numpy(Domain.from_numpy(x_), x_).to_sparse()

        pca = PCA(10, svd_solver="full", random_state=random_state)(x.to_dense())
        rpca = PCA(10, svd_solver="randomized", random_state=random_state)(x)

        np.testing.assert_almost_equal(
            pca.components_, rpca.components_, decimal=8
        )
        np.testing.assert_almost_equal(
            pca.explained_variance_, rpca.explained_variance_, decimal=8
        )
        np.testing.assert_almost_equal(
            pca.singular_values_, rpca.singular_values_, decimal=8
        )

        # And take a short, fat matrix
        x_ = random_state.negative_binomial(1, 0.5, (20, 100))
        x = Table.from_numpy(Domain.from_numpy(x_), x_).to_sparse()

        pca = PCA(10, svd_solver="full", random_state=random_state)(x.to_dense())
        rpca = PCA(10, svd_solver="randomized", random_state=random_state)(x)

        np.testing.assert_almost_equal(
            pca.components_, rpca.components_, decimal=8
        )
        np.testing.assert_almost_equal(
            pca.explained_variance_, rpca.explained_variance_, decimal=8
        )
        np.testing.assert_almost_equal(
            pca.singular_values_, rpca.singular_values_, decimal=8
        )

    def test_incremental_pca(self):
        data = self.ionosphere
        self.__ipca_test_helper(data, n_com=3, min_xpl_var=0.49)
        self.__ipca_test_helper(data, n_com=32, min_xpl_var=1)

    def __ipca_test_helper(self, data, n_com, min_xpl_var):
        pca = IncrementalPCA(n_components=n_com)
        pca_model = pca(data[::2])
        pca_xpl_var = np.sum(pca_model.explained_variance_ratio_)
        self.assertGreaterEqual(pca_xpl_var + 1e-6, min_xpl_var)
        self.assertEqual(n_com, pca_model.n_components)
        self.assertEqual((n_com, data.X.shape[1]), pca_model.components_.shape)
        proj = np.dot(data.X - pca_model.mean_, pca_model.components_.T)
        np.testing.assert_almost_equal(pca_model(data).X, proj)
        pc1_ipca = pca_model.components_[0]
        self.assertAlmostEqual(np.linalg.norm(pc1_ipca), 1)
        pc1_pca = PCA(n_components=n_com)(data).components_[0]
        self.assertAlmostEqual(np.linalg.norm(pc1_pca), 1)
        self.assertNotAlmostEqual(abs(pc1_ipca.dot(pc1_pca)), 1, 2)
        pc1_ipca = pca_model.partial_fit(data[1::2]).components_[0]
        self.assertAlmostEqual(abs(pc1_ipca.dot(pc1_pca)), 1, 4)

    def test_truncated_svd(self):
        data = self.ionosphere
        self.__truncated_svd_test_helper(data, n_components=3, min_variance=0.5)
        self.__truncated_svd_test_helper(data, n_components=10, min_variance=0.7)
        self.__truncated_svd_test_helper(data, n_components=31, min_variance=0.99)

    def __truncated_svd_test_helper(self, data, n_components, min_variance):
        model = TruncatedSVD(n_components=n_components)(data)
        svd_variance = np.sum(model.explained_variance_ratio_)
        self.assertGreaterEqual(svd_variance + 1e-6, min_variance)
        self.assertEqual(n_components, model.n_components)
        self.assertEqual((n_components, data.X.shape[1]), model.components_.shape)
        proj = np.dot(data.X, model.components_.T)
        np.testing.assert_almost_equal(model(data).X, proj)

    def test_compute_value(self):
        iris = self.iris
        pca = PCA(n_components=2)(iris)
        pca_iris = pca(iris)
        pca_iris2 = iris.transform(pca_iris.domain)
        np.testing.assert_almost_equal(pca_iris.X, pca_iris2.X)
        np.testing.assert_equal(pca_iris.Y, pca_iris2.Y)

        pca_iris3 = pickle.loads(pickle.dumps(pca_iris))
        np.testing.assert_almost_equal(pca_iris.X, pca_iris3.X)
        np.testing.assert_equal(pca_iris.Y, pca_iris3.Y)

    def test_transformed_domain_does_not_pickle_data(self):
        iris = self.iris
        pca = PCA(n_components=2)(iris)
        pca_iris = pca(iris)
        pca_iris2 = iris.transform(pca_iris.domain)

        pca_iris2 = pickle.loads(pickle.dumps(pca_iris))
        self.assertIsNone(pca_iris2.domain[0].compute_value.transformed)

    def test_chain(self):
        zoo_c = Continuize()(self.zoo)
        pca = PCA(n_components=3)(zoo_c)(self.zoo)
        pca2 = PCA(n_components=3)(zoo_c)(zoo_c)
        pp = [Continuize()]
        pca3 = PCA(n_components=3, preprocessors=pp)(self.zoo)(self.zoo)
        np.testing.assert_almost_equal(pca.X, pca2.X)
        np.testing.assert_almost_equal(pca.X, pca3.X)

    def test_PCA_scorer(self):
        data = self.iris
        pca = PCA(preprocessors=[Normalize()])
        pca.component = 1
        scores = pca.score_data(data)
        self.assertEqual(scores.shape[1], len(data.domain.attributes))
        self.assertEqual(['petal length', 'petal width'],
                         sorted([data.domain.attributes[i].name
                                 for i in np.argsort(scores[0])[-2:]]))
        self.assertEqual([round(s, 4) for s in scores[0]],
                         [0.5224, 0.2634, 0.5813, 0.5656])

    def test_PCA_scorer_component(self):
        pca = PCA()
        for i in range(1, len(self.zoo.domain.attributes) + 1):
            pca.component = i
            scores = pca.score_data(self.zoo)
            self.assertEqual(scores.shape,
                             (pca.component, len(self.zoo.domain.attributes)))

    def test_PCA_scorer_all_components(self):
        n_attr = len(self.iris.domain.attributes)
        pca = PCA()
        scores = pca.score_data(self.iris)
        self.assertEqual(scores.shape, (n_attr, n_attr))

    def test_max_components(self):
        d = np.random.RandomState(0).rand(20, 20)
        data = Table.from_numpy(None, d)
        pca = PCA()(data)
        self.assertEqual(len(pca.explained_variance_ratio_), 20)
        pca = PCA(n_components=10)(data)
        self.assertEqual(len(pca.explained_variance_ratio_), 10)

    def test_eq_hash(self):
        d = np.random.RandomState(0).rand(20, 20)
        data = Table.from_numpy(None, d)
        p1 = PCA()(data)
        p2 = PCA()(data)
        np.testing.assert_equal(p1(data).X, p2(data).X)

        # even though results are the same, these transformations
        # are different because the PCA object is
        self.assertNotEqual(p1, p2)
        self.assertNotEqual(p1.domain, p2.domain)
        self.assertNotEqual(hash(p1), hash(p2))
        self.assertNotEqual(hash(p1.domain), hash(p2.domain))

        # copy projection
        p2.domain[0].compute_value.compute_shared.projection = \
            p1.domain[0].compute_value.compute_shared.projection
        p2.proj = p1.proj
        # reset hash caches because object were hacked
        # pylint: disable=protected-access
        p1.domain._hash = None
        p2.domain._hash = None
        p1.domain[0].compute_value.compute_shared._hash = None
        p2.domain[0].compute_value.compute_shared._hash = None

        self.assertEqual(p1, p2)
        self.assertEqual(p1.domain, p2.domain)
        self.assertEqual(hash(p1), hash(p2))
        self.assertEqual(hash(p1.domain), hash(p2.domain))
