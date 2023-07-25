# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring

import unittest

import numpy as np
from sklearn import linear_model

from Orange.data import Table
from Orange.regression import (LinearRegressionLearner,
                               RidgeRegressionLearner,
                               LassoRegressionLearner,
                               ElasticNetLearner,
                               ElasticNetCVLearner,
                               MeanLearner)
from Orange.evaluation import CrossValidation, RMSE
from Orange.tests.test_dasktable import with_dasktable, temp_dasktable


class TestLinearRegressionLearner(unittest.TestCase):
    learners = [
        RidgeRegressionLearner(),
        LassoRegressionLearner(),
        ElasticNetLearner(),
        ElasticNetCVLearner(),
        MeanLearner()
    ]

    @classmethod
    def setUpClass(cls):
        cls.housing = Table("housing")

    @with_dasktable
    def test_LinearRegression(self, prepare_table):
        nrows = 1000
        ncols = 3
        x = np.random.randint(-20, 51, (nrows, ncols))
        c = np.random.rand(ncols, 1) * 10 - 3
        e = np.random.rand(nrows, 1) - 0.5
        y = np.dot(x, c) + e

        x1, x2 = np.split(x, 2)
        y1, y2 = np.split(y, 2)
        t = prepare_table(Table.from_numpy(None, x1, y1))
        learn = LinearRegressionLearner()
        clf = learn(t)
        z = clf(x2)
        self.assertTrue((abs(z.reshape(-1, 1) - y2) < 2.0).all())

    def test_Regression(self):
        cv = CrossValidation(k=2)
        res = cv(self.housing, self.learners)
        rmse = RMSE(res)
        for i in range(len(self.learners) - 1):
            self.assertLess(rmse[i], rmse[-1])

    def test_linear_scorer(self):
        learner = LinearRegressionLearner()
        scores = learner.score_data(self.housing)
        self.assertEqual(
            'LSTAT', self.housing.domain.attributes[np.argmax(scores[0])].name)
        self.assertEqual(scores.shape[1], len(self.housing.domain.attributes))

    def test_scorer(self):
        learners = [LinearRegressionLearner(),
                    RidgeRegressionLearner(),
                    LassoRegressionLearner(alpha=0.01),
                    ElasticNetLearner(alpha=0.01)]
        for learner in learners:
            scores = learner.score_data(self.housing)
            self.assertEqual(
                'LSTAT',
                self.housing.domain.attributes[np.argmax(scores[0])].name)
            self.assertEqual(scores.shape[1],
                             len(self.housing.domain.attributes))

    def test_scorer_feature(self):
        learners = [LinearRegressionLearner(),
                    RidgeRegressionLearner(),
                    LassoRegressionLearner(alpha=0.01),
                    ElasticNetLearner(alpha=0.01)]
        for learner in learners:
            scores = learner.score_data(self.housing)
            for i, attr in enumerate(self.housing.domain.attributes):
                score = learner.score_data(self.housing, attr)
                np.testing.assert_array_almost_equal(score, scores[:, i])

    def test_coefficients(self):
        data = Table.from_numpy(None, [[11], [12], [13]], [0, 1, 2])
        model = LinearRegressionLearner()(data)
        self.assertAlmostEqual(float(model.intercept), -11)
        self.assertEqual(len(model.coefficients), 1)
        self.assertAlmostEqual(float(model.coefficients[0]), 1)

    def test_comparison_with_sklearn(self):
        alphas = [0.001, 0.1, 1, 10, 100]
        learners = [(LassoRegressionLearner, linear_model.Lasso),
                    (RidgeRegressionLearner, linear_model.Ridge),
                    (ElasticNetLearner, linear_model.ElasticNet)]
        for o_learner, s_learner in learners:
            for a in alphas:
                lr = o_learner(alpha=a)
                o_model = lr(self.housing)
                s_model = s_learner(alpha=a, fit_intercept=True)
                s_model.fit(self.housing.X, self.housing.Y)
                delta = np.sum(s_model.coef_ - o_model.coefficients)
                self.assertAlmostEqual(delta, 0.0)

    def test_comparison_elastic_net(self):
        alphas = [0.001, 0.1, 1, 10, 100]
        for a in alphas:
            lasso = LassoRegressionLearner(alpha=a)
            lasso_model = lasso(self.housing)
            en = ElasticNetLearner(alpha=a, l1_ratio=1)
            en_model = en(self.housing)
            np.testing.assert_allclose(
                lasso_model.coefficients, en_model.coefficients, atol=a/10)

    def test_linear_regression_repr(self):
        learner = LinearRegressionLearner()
        repr_text = repr(learner)
        learner2 = eval(repr_text)

        self.assertIsInstance(learner2, LinearRegressionLearner)


class TestLinearRegressionLearnerOnDask(TestLinearRegressionLearner):
    learners = [
        RidgeRegressionLearner(),
        LassoRegressionLearner(),
        ElasticNetLearner(),
        MeanLearner()
    ]

    @classmethod
    def setUpClass(cls):
        cls.housing = temp_dasktable(Table("housing"))

    @unittest.skip("already tested")
    def test_LinearRegression(self, _):
        super().test_LinearRegression(_)

    @unittest.skip("scores differ from sklearn")
    def test_comparison_with_sklearn(self):
        super().test_comparison_with_sklearn()
