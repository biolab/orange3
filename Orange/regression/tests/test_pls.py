# pylint: disable=missing-docstring
import unittest

import numpy as np
from sklearn.cross_decomposition import PLSRegression

from Orange.data import Table, Domain, ContinuousVariable
from Orange.regression import PLSRegressionLearner
from Orange.regression.pls import _PLSCommonTransform


def table(rows, attr, variables):
    attr_vars = [ContinuousVariable(name=f"Feature {i}") for i in
                 range(attr)]
    class_vars = [ContinuousVariable(name=f"Class {i}") for i in
                  range(variables)]
    domain = Domain(attr_vars, class_vars, [])
    X = np.random.RandomState(0).random((rows, attr))
    Y = np.random.RandomState(1).random((rows, variables))
    return Table.from_numpy(domain, X=X, Y=Y)


class TestPLSRegressionLearner(unittest.TestCase):
    def test_fitted_parameters(self):
        fitted_parameters = PLSRegressionLearner().fitted_parameters
        self.assertIsInstance(fitted_parameters, list)
        self.assertEqual(len(fitted_parameters), 1)

    def test_allow_y_dim(self):
        """ The current PLS version allows only a single Y dimension. """
        learner = PLSRegressionLearner(n_components=2)
        d = table(10, 5, 0)
        with self.assertRaises(ValueError):
            learner(d)
        for n_class_vars in [1, 2, 3]:
            d = table(10, 5, n_class_vars)
            learner(d)  # no exception

    def test_compare_to_sklearn(self):
        d = table(10, 5, 1)
        orange_model = PLSRegressionLearner()(d)
        scikit_model = PLSRegression().fit(d.X, d.Y)
        np.testing.assert_almost_equal(scikit_model.predict(d.X).ravel(),
                                       orange_model(d))
        np.testing.assert_almost_equal(scikit_model.coef_,
                                       orange_model.coefficients)

    def test_compare_to_sklearn_multid(self):
        d = table(10, 5, 3)
        orange_model = PLSRegressionLearner()(d)
        scikit_model = PLSRegression().fit(d.X, d.Y)
        np.testing.assert_almost_equal(scikit_model.predict(d.X),
                                       orange_model(d))
        np.testing.assert_almost_equal(scikit_model.coef_,
                                       orange_model.coefficients)

    def test_too_many_components(self):
        # do not change n_components
        d = table(5, 5, 1)
        model = PLSRegressionLearner(n_components=4)(d)
        self.assertEqual(model.skl_model.n_components, 4)
        # need to use fewer components; column limited
        d = table(6, 5, 1)
        model = PLSRegressionLearner(n_components=6)(d)
        self.assertEqual(model.skl_model.n_components, 4)
        # need to use fewer components; row limited
        d = table(5, 6, 1)
        model = PLSRegressionLearner(n_components=6)(d)
        self.assertEqual(model.skl_model.n_components, 4)

    def test_scores(self):
        for d in [table(10, 5, 1), table(10, 5, 3)]:
            orange_model = PLSRegressionLearner()(d)
            scikit_model = PLSRegression().fit(d.X, d.Y)
            scores = orange_model.project(d)
            sx, sy = scikit_model.transform(d.X, d.Y)
            np.testing.assert_almost_equal(sx, scores.X)
            np.testing.assert_almost_equal(sy, scores.metas)

    def test_components(self):
        def t2d(m):
            return m.reshape(-1, 1) if len(m.shape) == 1 else m

        for d in [table(10, 5, 1), table(10, 5, 3)]:
            orange_model = PLSRegressionLearner()(d)
            scikit_model = PLSRegression().fit(d.X, d.Y)
            components = orange_model.components()
            np.testing.assert_almost_equal(scikit_model.x_loadings_,
                                           components.X.T)
            np.testing.assert_almost_equal(scikit_model.y_loadings_,
                                           t2d(components.Y).T)

    def test_coefficients(self):
        for d in [table(10, 5, 1), table(10, 5, 3)]:
            orange_model = PLSRegressionLearner()(d)
            scikit_model = PLSRegression().fit(d.X, d.Y)
            coef_table = orange_model.coefficients_table()
            np.testing.assert_almost_equal(scikit_model.coef_.T,
                                           coef_table.X)

    def test_residuals_normal_probability(self):
        for d in [table(10, 5, 1), table(10, 5, 3)]:
            orange_model = PLSRegressionLearner()(d)
            res_table = orange_model.residuals_normal_probability(d)
            n_target = len(d.domain.class_vars)
            self.assertEqual(res_table.X.shape, (len(d), 2 * n_target))

    def test_dmodx(self):
        for d in (table(10, 5, 1), table(10, 5, 3)):
            orange_model = PLSRegressionLearner()(d)
            dist_table = orange_model.dmodx(d)
            self.assertEqual(dist_table.X.shape, (len(d), 1))

    def test_eq_hash(self):
        data = Table("housing")
        pls1 = PLSRegressionLearner()(data)
        pls2 = PLSRegressionLearner()(data)

        proj1 = pls1.project(data)
        proj2 = pls2.project(data)

        np.testing.assert_equal(proj1.X, proj2.X)
        np.testing.assert_equal(proj1.metas, proj2.metas)

        # even though results are the same, these transformations
        # are different because the PLS object is
        self.assertNotEqual(proj1, proj2)
        self.assertNotEqual(proj1.domain, proj2.domain)
        self.assertNotEqual(hash(proj1), hash(proj2))
        self.assertNotEqual(hash(proj1.domain), hash(proj2.domain))

    def test_eq_hash_fake_same_model(self):
        data = Table("housing")
        pls1 = PLSRegressionLearner()(data)
        pls2 = PLSRegressionLearner()(data)

        proj1 = pls1.project(data)
        proj2 = pls2.project(data)

        proj2.domain[0].compute_value.compute_shared.pls_model = \
            proj1.domain[0].compute_value.compute_shared.pls_model
        # reset hash caches because object were hacked
        # pylint: disable=protected-access
        proj1.domain._hash = None
        proj2.domain._hash = None

        self.assertEqual(proj1.domain, proj2.domain)
        self.assertEqual(hash(proj1.domain), hash(proj2.domain))


class TestPLSCommonTransform(unittest.TestCase):
    def test_eq(self):
        m = PLSRegressionLearner()(table(10, 5, 1))
        transformer = _PLSCommonTransform(m)
        self.assertEqual(transformer, transformer)
        self.assertEqual(transformer, _PLSCommonTransform(m))

        m = PLSRegressionLearner()(table(10, 5, 2))
        self.assertNotEqual(transformer, _PLSCommonTransform(m))

    def test_hash(self):
        m = PLSRegressionLearner()(table(10, 5, 1))
        transformer = _PLSCommonTransform(m)
        self.assertEqual(hash(transformer), hash(transformer))
        self.assertEqual(hash(transformer), hash(_PLSCommonTransform(m)))

        m = PLSRegressionLearner()(table(10, 5, 2))
        self.assertNotEqual(hash(transformer), hash(_PLSCommonTransform(m)))

    def test_missing_target(self):
        data = table(10, 5, 1)
        with data.unlocked(data.Y):
            data.Y[::3] = np.nan
        pls = PLSRegressionLearner()(data)
        proj = pls.project(data)
        self.assertFalse(np.isnan(proj.X).any())
        self.assertFalse(np.isnan(proj.metas[1::3]).any())
        self.assertFalse(np.isnan(proj.metas[2::3]).any())
        self.assertTrue(np.isnan(proj.metas[::3]).all())

    def test_missing_target_multitarget(self):
        data = table(10, 5, 3)
        with data.unlocked(data.Y):
            data.Y[0] = np.nan
            data.Y[1, 1] = np.nan

        pls = PLSRegressionLearner()(data)
        proj = pls.project(data)
        self.assertFalse(np.isnan(proj.X).any())
        self.assertFalse(np.isnan(proj.metas[2:]).any())
        self.assertTrue(np.isnan(proj.metas[:2]).all())

    def test_apply_domain_classless_data(self):
        data = Table("housing")
        pls = PLSRegressionLearner()(data)
        classless_data = data.transform(Domain(data.domain.attributes))[:5]

        proj = pls.project(classless_data)
        self.assertFalse(np.isnan(proj.X).any())
        self.assertTrue(np.isnan(proj.metas).all())


if __name__ == "__main__":
    unittest.main()
