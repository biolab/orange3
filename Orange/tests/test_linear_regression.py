import unittest
import numpy as np
from Orange.data import Table
from Orange.regression import (LinearRegressionLearner,
                               RidgeRegressionLearner,
                               LassoRegressionLearner,
                               ElasticNetLearner,
                               ElasticNetCVLearner,
                               MeanLearner)
from Orange.evaluation import CrossValidation, RMSE


class LinearRegressionTest(unittest.TestCase):
    def test_LinearRegression(self):
        nrows = 1000
        ncols = 3
        x = np.random.random_integers(-20, 50, (nrows, ncols))
        c = np.random.rand(ncols, 1) * 10 - 3
        e = np.random.rand(nrows, 1) - 0.5
        y = np.dot(x, c) + e

        x1, x2 = np.split(x, 2)
        y1, y2 = np.split(y, 2)
        t = Table(x1, y1)
        learn = LinearRegressionLearner()
        clf = learn(t)
        z = clf(x2)
        self.assertTrue((abs(z.reshape(-1, 1) - y2) < 2.0).all())

    def test_Regression(self):
        data = Table("housing")
        ridge = RidgeRegressionLearner()
        lasso = LassoRegressionLearner()
        elastic = ElasticNetLearner()
        elasticCV = ElasticNetCVLearner()
        mean = MeanLearner()
        learners = [ridge, lasso, elastic, elasticCV, mean]
        res = CrossValidation(data, learners, k=2)
        rmse = RMSE(res)
        for i in range(len(learners) - 1):
            self.assertTrue(rmse[i] < rmse[-1])

    def test_linear_scorer(self):
        data = Table('housing')
        learner = LinearRegressionLearner()
        scores = learner.score_data(data)
        self.assertEqual(len(scores), len(data.domain.attributes))

    def test_ridge_scorer(self):
        data = Table('housing')
        learner = RidgeRegressionLearner()
        scores = learner.score_data(data)
        self.assertEqual(len(scores), len(data.domain.attributes))

    def test_lasso_scorer(self):
        data = Table('housing')
        learner = LassoRegressionLearner()
        scores = learner.score_data(data)
        self.assertEqual(len(scores), len(data.domain.attributes))

    def test_linear_scorer_feature(self):
        data = Table('housing')
        learner = LinearRegressionLearner()
        scores = learner.score_data(data)
        for i, attr in enumerate(data.domain.attributes):
            score = learner.score_data(data, attr)
            self.assertEqual(score, scores[i])

    def test_ridge_scorer_feature(self):
        data = Table('housing')
        learner = RidgeRegressionLearner()
        scores = learner.score_data(data)
        for i, attr in enumerate(data.domain.attributes):
            score = learner.score_data(data, attr)
            self.assertEqual(score, scores[i])

    def test_lasso_scorer_feature(self):
        data = Table('housing')
        learner = LassoRegressionLearner()
        scores = learner.score_data(data)
        for i, attr in enumerate(data.domain.attributes):
            score = learner.score_data(data, attr)
            self.assertEqual(score, scores[i])
