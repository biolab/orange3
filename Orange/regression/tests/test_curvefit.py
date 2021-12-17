import pickle
import copy
import ast
import unittest

import numpy as np

from Orange.base import Model
from Orange.data import Table, Domain
from Orange.evaluation import CrossValidation, RMSE
from Orange.preprocess import Impute
from Orange.preprocess.impute import Random
from Orange.regression import CurveFitLearner
from Orange.regression.curvefit import _create_lambda
import Orange.tests

class TestCreateLambda(unittest.TestCase):
    def test_create_lambda_simple(self):
        func_, params_, vars_ = _create_lambda("a + b", [], [])
        self.assertTrue(callable(func_))
        self.assertEqual(params_, ["a", "b"])
        self.assertEqual(vars_, [])
        self.assertEqual(func_(np.array([[1, 11], [2, 22]]), 1, 2), 3)

    def test_create_lambda_var(self):
        func_, params_, vars_ = _create_lambda("var + a + b", ["var"], [])
        self.assertTrue(callable(func_))
        self.assertEqual(params_, ["a", "b"])
        self.assertEqual(vars_, ["var"])
        np.testing.assert_array_equal(
            func_(np.array([[1, 11], [2, 22]]), 1, 2),
            np.array([4, 5])
        )

    def test_create_lambda_fun(self):
        func_, params_, vars_ = _create_lambda("power(a, 2)", [], ["power"])
        self.assertTrue(callable(func_))
        self.assertEqual(params_, ["a"])
        self.assertEqual(vars_, [])
        np.testing.assert_array_equal(
            func_(np.array([[1, 11], [2, 22]]), 3),
            np.array([9, 9])
        )

    def test_create_lambda_var_fun(self):
        func_, params_, vars_ = _create_lambda(
            "var1 + power(a, 2) + power(a, 2)", ["var1", "var2"], ["power"]
        )
        self.assertTrue(callable(func_))
        self.assertEqual(params_, ["a"])
        self.assertEqual(vars_, ["var1"])
        np.testing.assert_array_equal(
            func_(np.array([[1, 11], [2, 22]]), 3),
            np.array([19, 20])
        )

    def test_create_lambda_x(self):
        func_, params_, vars_ = _create_lambda(
            "var1 + x", ["var1", "var2"], []
        )
        self.assertTrue(callable(func_))
        self.assertEqual(params_, ["x"])
        self.assertEqual(vars_, ["var1"])
        np.testing.assert_array_equal(
            func_(np.array([[1, 11], [2, 22]]), 3), np.array([4, 5])
        )

    def test_create_lambda_ast(self):
        func_, params_, vars_ = _create_lambda(
            ast.parse("a + b", mode="eval"), [], []
        )
        self.assertTrue(callable(func_))
        self.assertEqual(params_, ["a", "b"])
        self.assertEqual(vars_, [])
        self.assertEqual(func_(np.array([[1, 11], [2, 22]]), 1, 2), 3)

    def test_create_lambda(self):
        func_, params_, vars_ = _create_lambda(
            "a * var1 + b * exp(var2 * power(pi, 0))",
            ["var1", "var2", "var3"], ["exp", "power", "pi"]
        )
        self.assertTrue(callable(func_))
        self.assertEqual(params_, ["a", "b"])
        self.assertEqual(vars_, ["var1", "var2"])
        np.testing.assert_allclose(
            func_(np.array([[1, 2], [3, 4]]), 3, 2),
            np.array([17.778112, 118.1963])
        )


def func(x, a, b, c):
    return a * np.exp(-b * x[:, 0]) + c


class TestCurveFitLearner(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = Table("housing")

    def test_init_str(self):
        kw = dict(available_feature_names=[], functions=[])
        learner = CurveFitLearner("a + b", **kw)
        self.assertIsInstance(learner, CurveFitLearner)

        self.assertRaises(TypeError, CurveFitLearner, "a + b")

        kw = dict(available_feature_names=[])
        self.assertRaises(TypeError, CurveFitLearner, "a + b", **kw)

    def test_init_ast(self):
        kw = dict(available_feature_names=[], functions=[])
        exp = ast.parse("a + b", mode="eval")
        learner = CurveFitLearner(exp, **kw)
        self.assertIsInstance(learner, CurveFitLearner)

        self.assertRaises(TypeError, CurveFitLearner, exp)

    def test_init_callable(self):
        kw = dict(parameters_names=[], features_names=[])
        learner = CurveFitLearner(lambda x, a: a, **kw)
        self.assertIsInstance(learner, CurveFitLearner)

        self.assertRaises(TypeError, CurveFitLearner, lambda x, a: a)

        kw = dict(parameters_names=[])
        self.assertRaises(TypeError, CurveFitLearner, lambda x, a: a, **kw)

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
        with data.unlocked():
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

    # pylint: disable=unsubscriptable-object
    def test_cv_preprocess(self):
        def fun(x, a):
            return x[:, 0] + a

        imputer = Impute()
        learner = CurveFitLearner(fun, ["a"], ["CRIM"])
        cv = CrossValidation(k=2)
        results = cv(self.data, [learner])
        rmse1 = RMSE(results)[0]

        learner = CurveFitLearner(fun, ["a"], ["CRIM"])
        cv = CrossValidation(k=2)
        results = cv(self.data, [learner], preprocessor=imputer)
        rmse2 = RMSE(results)[0]

        learner = CurveFitLearner(fun, ["a"], ["CRIM"], preprocessors=imputer)
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

    def test_can_copy_str(self):
        available_feature_names = [a.name for a in self.data.domain.attributes]
        learner = CurveFitLearner(
            "a * exp(-b * CRIM) + c",
            available_feature_names=available_feature_names,
            functions=["exp"],
        )

        model = learner(self.data)
        pred = model(self.data)

        np.testing.assert_array_equal(
            pred, copy.deepcopy(model)(self.data)
        )
        np.testing.assert_array_equal(
            pred, copy.deepcopy(learner)(self.data)(self.data)
        )

    def test_can_copy_callable(self):
        learner = CurveFitLearner(func, [], ["CRIM"])
        self.assertRaises(AttributeError, copy.deepcopy, learner)
        self.assertRaises(AttributeError, copy.deepcopy, learner(self.data))

    def test_can_copy_with_imputer(self):
        available_feature_names = [a.name for a in self.data.domain.attributes]
        learner = CurveFitLearner(
            "a * exp(-b * CRIM) + c",
            available_feature_names=available_feature_names,
            functions=["exp"],
            preprocessors=Impute()
        )
        copy.deepcopy(learner)
        copy.deepcopy(learner(self.data))

        learner = CurveFitLearner(
            "a * exp(-b * CRIM) + c",
            available_feature_names=available_feature_names,
            functions=["exp"],
            preprocessors=Impute(method=Random())
        )
        copy.deepcopy(learner)
        # uncomment when issue #5480 is solved
        # copy.deepcopy(learner(self.data))

    def test_can_pickle_str(self):
        available_feature_names = [a.name for a in self.data.domain.attributes]
        learner = CurveFitLearner(
            "a * exp(-b * CRIM) + c",
            available_feature_names=available_feature_names,
            functions=["exp"],
        )

        model = learner(self.data)

        dumped_learner = pickle.dumps(learner)
        loaded_learner = pickle.loads(dumped_learner)

        dumped_model = pickle.dumps(model)
        loaded_model = pickle.loads(dumped_model)

        np.testing.assert_array_equal(model(self.data),
                                      loaded_model(self.data))
        np.testing.assert_array_equal(model(self.data),
                                      loaded_learner(self.data)(self.data))

    def test_can_pickle_callable(self):
        learner = CurveFitLearner(
            lambda x, a, b, c: a * np.exp(-b * x[:, 0]) + c, [], ["CRIM"]
        )
        self.assertRaises(AttributeError, pickle.dumps, learner)
        self.assertRaises(AttributeError, pickle.dumps, learner(self.data))


if __name__ == "__main__":
    unittest.main()
