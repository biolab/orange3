import pickle
import copy
import unittest

import numpy as np

from Orange.base import Model
from Orange.data import Table, Domain
from Orange.evaluation import CrossValidation, RMSE
from Orange.preprocess import Impute
from Orange.preprocess.impute import Random
from Orange.regression import CurveFitLearner


def func(x, a, b, c):
    return a * np.exp(-b * x[:, 0]) + c


class TestCurveFitLearner(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = Table("housing")

    def test_function(self):
        CurveFitLearner(func, [], ["CRIM"])
        self.assertRaises(TypeError, CurveFitLearner, "y=a+b")

    def test_fit(self):
        learner = CurveFitLearner(func, [], ["CRIM"])
        model = learner(self.data)
        self.assertIsInstance(model, Model)

    def test_fit_no_params(self):
        learner = CurveFitLearner(lambda x: x[:, 0] + 1, [], ["CRIM"])
        self.assertRaises(ValueError, learner, self.data)

    def test_predict(self):
        learner = CurveFitLearner(func, [], ["CRIM"])
        model = learner(self.data)
        pred = model(self.data)
        self.assertEqual(len(pred), len(self.data))

    def test_predict_constant(self):
        def constant(_, a):
            return a

        learner = CurveFitLearner(constant, [], ["CRIM"])
        model = learner(self.data)
        pred = model(self.data)
        print(pred.shape)
        self.assertEqual(pred.shape, (len(self.data),))

    def test_coefficients(self):
        learner = CurveFitLearner(func, ["a", "b", "c"], ["LSTAT"])
        model = learner(self.data)
        coef = model.coefficients
        self.assertEqual(len(coef), 3)
        self.assertEqual(len(coef.domain.variables), 1)
        self.assertEqual(len(coef.domain.metas), 1)

    def test_inadequate_data(self):
        data = Table("iris")
        learner = CurveFitLearner(func, [], ["sepal length"])
        self.assertRaises(ValueError, learner, data)

        learner = CurveFitLearner(func, [], ["iris"])
        attributes = data.domain.attributes[:-1]
        class_var = data.domain.attributes[-1]
        domain = Domain(attributes + data.domain.class_vars, class_var)
        self.assertRaises(ValueError, learner, data.transform(domain))

    def test_missing_values(self):
        data = self.data.copy()
        data.X[0, 12] = np.nan
        learner = CurveFitLearner(func, [], ["CRIM"])
        model = learner(data)
        pred = model(data)
        self.assertEqual(len(pred), len(data))

    def test_cv(self):
        learner = CurveFitLearner(func, [], ["CRIM"])
        cv = CrossValidation(k=10)
        results = cv(self.data, [learner])
        RMSE(results)

    def test_cv_preprocess(self):
        def f(x, a):
            return x[:, 0] + a

        imputer = Impute()
        learner = CurveFitLearner(f, ["a"], ["CRIM"])
        cv = CrossValidation(k=2)
        results = cv(self.data, [learner])
        rmse1 = RMSE(results)[0]

        learner = CurveFitLearner(f, ["a"], ["CRIM"])
        cv = CrossValidation(k=2)
        results = cv(self.data, [learner], preprocessor=imputer)
        rmse2 = RMSE(results)[0]

        learner = CurveFitLearner(f, ["a"], ["CRIM"], preprocessors=imputer)
        cv = CrossValidation(k=2)
        results = cv(self.data, [learner])
        rmse3 = RMSE(results)[0]

        self.assertEqual(rmse1, rmse2)
        self.assertEqual(rmse2, rmse3)

    def test_predict_single_instance(self):
        learner = CurveFitLearner(func, [], ["CRIM"])
        model = learner(self.data)
        for ins in self.data:
            pred = model(ins)
            self.assertGreater(pred, 0)

    def test_predict_table(self):
        learner = CurveFitLearner(func, [], ["CRIM"])
        model = learner(self.data)
        pred = model(self.data)
        self.assertEqual(pred.shape, (len(self.data),))
        self.assertGreater(all(pred), 0)

    def test_predict_numpy(self):
        learner = CurveFitLearner(func, [], ["CRIM"])
        model = learner(self.data)
        pred = model(self.data.X)
        self.assertEqual(pred.shape, (len(self.data),))
        self.assertGreater(all(pred), 0)

    def test_predict_sparse(self):
        sparse_data = self.data.to_sparse()
        learner = CurveFitLearner(func, [], ["CRIM"])
        self.assertRaises(TypeError, learner, sparse_data)

    def test_can_copy(self):
        learner = CurveFitLearner(func, [], ["CRIM"])
        model = learner(self.data)
        pred = model(self.data)
        np.testing.assert_array_equal(
            pred, copy.deepcopy(model)(self.data)
        )
        np.testing.assert_array_equal(
            pred, copy.deepcopy(learner)(self.data)(self.data)
        )

    def test_can_copy_with_imputer(self):
        learner = CurveFitLearner(
            func, [], ["CRIM"], preprocessors=Impute()
        )
        copy.deepcopy(learner)
        learner = CurveFitLearner(
            func, [], ["CRIM"], preprocessors=Impute(method=Random())
        )
        copy.deepcopy(learner)

    def test_can_pickle(self):
        learner = CurveFitLearner(
            lambda x, a, b, c: a * np.exp(-b * x[:, 0]) + c, [], ["CRIM"]
        )

        model = learner(self.data)
        pred = model(self.data)

        loaded_model = pickle.loads(pickle.dumps(model))
        loaded_pred = loaded_model(self.data)

        np.testing.assert_array_equal(pred, loaded_pred)


if __name__ == "__main__":
    unittest.main()
