import unittest
import numpy as np
import Orange

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
        t = Orange.data.Table(x1, y1)
        learn = Orange.regression.LinearRegressionLearner()
        clf = learn(t)
        z = clf(x2)
        self.assertTrue((abs(z.reshape(-1, 1) - y2) < 2.0).all())

    def test_Regression(self):
        data = Orange.data.Table("housing")
        ridge = Orange.regression.RidgeRegressionLearner()
        lasso = Orange.regression.LassoRegressionLearner()
        elastic = Orange.regression.ElasticNetLearner()
        elasticCV = Orange.regression.ElasticNetCVLearner()
        mean = Orange.regression.MeanLearner()
        learners = [ridge, lasso, elastic, elasticCV, mean]
        res = Orange.evaluation.CrossValidation(data, learners, k=2)
        rmse = Orange.evaluation.RMSE(res)
        for i in range(len(learners)-1):
            self.assertTrue(rmse[i] < rmse[-1])
