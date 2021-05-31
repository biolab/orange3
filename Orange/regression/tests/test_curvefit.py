import unittest

import numpy as np

from Orange.base import Model
from Orange.data import Table, Domain
from Orange.evaluation import CrossValidation, RMSE
from Orange.regression import CurveFitLearner


def func(x, a, b, c):
    return a * np.exp(-b * x[:, 12]) + c


class TestCurveFitLearner(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = Table("housing")

    def test_function(self):
        CurveFitLearner(func, [])
        self.assertRaises(TypeError, CurveFitLearner, "y=a+b")

    def test_fit(self):
        learner = CurveFitLearner(func, [])
        model = learner(self.data)
        self.assertIsInstance(model, Model)

    def test_fit_no_params(self):
        learner = CurveFitLearner(lambda x: x[:, 0] + 1, [])
        self.assertRaises(ValueError, learner, self.data)

    def test_predict(self):
        learner = CurveFitLearner(func, [])
        model = learner(self.data)
        pred = model(self.data)
        self.assertEqual(len(pred), len(self.data))

    def test_coefficients(self):
        learner = CurveFitLearner(func, ["a", "b", "c"])
        model = learner(self.data)
        coef = model.coefficients
        self.assertEqual(len(coef), 3)
        self.assertEqual(len(coef.domain.variables), 1)
        self.assertEqual(len(coef.domain.metas), 1)

    def test_inadequate_data(self):
        data = Table("iris")
        learner = CurveFitLearner(func, [])
        self.assertRaises(ValueError, learner, data)

        learner = CurveFitLearner(func, [])
        attributes = data.domain.attributes[:-1]
        class_var = data.domain.attributes[-1]
        domain = Domain(attributes + data.domain.class_vars, class_var)
        self.assertRaises(ValueError, learner, data.transform(domain))

    def test_missing_values(self):
        data = self.data.copy()
        data.X[0, 12] = np.nan
        learner = CurveFitLearner(func, [])
        model = learner(data)
        pred = model(data)
        self.assertEqual(len(pred), len(data))

    def test_cv(self):
        learner = CurveFitLearner(func, [])
        cv = CrossValidation(k=10)
        results = cv(self.data, [learner])
        RMSE(results)

    def test_predict_single_instance(self):
        learner = CurveFitLearner(func, [])
        model = learner(self.data)
        for ins in self.data:
            pred = model(ins)
            self.assertGreater(pred, 0)

    def test_predict_table(self):
        learner = CurveFitLearner(func, [])
        model = learner(self.data)
        pred = model(self.data)
        self.assertEqual(pred.shape, (len(self.data),))
        self.assertGreater(all(pred), 0)

    def test_predict_numpy(self):
        learner = CurveFitLearner(func, [])
        model = learner(self.data)
        pred = model(self.data.X)
        self.assertEqual(pred.shape, (len(self.data),))
        self.assertGreater(all(pred), 0)

    def test_predict_sparse(self):
        sparse_data = self.data.to_sparse()
        learner = CurveFitLearner(func, [])
        self.assertRaises(TypeError, learner, sparse_data)


if __name__ == "__main__":
    unittest.main()
