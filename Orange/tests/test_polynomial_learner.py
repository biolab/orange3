import unittest
import numpy as np
import Orange

class PolynomialLearnerTest(unittest.TestCase):
    def test_PolynomialLearner(self):
        x = np.array([0.172, 0.167, 0.337, 0.420, 0.355, 0.710, 0.801, 0.876])
        y = np.array([0.784, 0.746, 0.345, 0.363, 0.366, 0.833, 0.490, 0.445])
        
        data = Orange.data.Table(x.reshape(-1,1), y)
        data.domain = Orange.data.Domain([Orange.data.ContinuousVariable('x')], 
            class_vars=[Orange.data.ContinuousVariable('y')])

        linear = Orange.regression.LinearRegressionLearner()
        polynomial2 = Orange.regression.PolynomialLearner(linear, degree=2)
        polynomial3 = Orange.regression.PolynomialLearner(linear, degree=3)

        res = Orange.evaluation.TestOnTrainingData(data, 
            [linear, polynomial2, polynomial3])
        rmse = Orange.evaluation.RMSE(res)

        self.assertTrue(rmse[1] < rmse[0])
        self.assertTrue(rmse[2] < rmse[1])
