import unittest
from typing import Callable

from Orange.base import XGBBase
from Orange.data import Table
from Orange.evaluation import CrossValidation, RMSE
from Orange.preprocess.score import Scorer

try:
    from Orange.regression import XGBRegressor, XGBRFRegressor
except ImportError:
    XGBRegressor = XGBRFRegressor = None


def test_learners(func: Callable) -> Callable:
    def wrapper(self):
        func(self, XGBRegressor)
        func(self, XGBRFRegressor)

    return wrapper


@unittest.skipIf(XGBRegressor is None, "Missing 'xgboost' package")
class TestXGBReg(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.housing = Table("housing")

    @test_learners
    def test_XGB(self, learner_class: XGBBase):
        booster = learner_class()
        cv = CrossValidation(k=10)
        results = cv(self.housing, [booster])
        RMSE(results)

    @test_learners
    def test_predict_single_instance(self, learner_class: XGBBase):
        booster = learner_class()
        model = booster(self.housing)
        for ins in self.housing:
            pred = model(ins)
            self.assertGreater(pred, 0)

    @test_learners
    def test_predict_table(self, learner_class: XGBBase):
        booster = learner_class()
        model = booster(self.housing)
        pred = model(self.housing)
        self.assertEqual(len(self.housing), len(pred))
        self.assertGreater(all(pred), 0)

    @test_learners
    def test_predict_numpy(self, learner_class: XGBBase):
        booster = learner_class()
        model = booster(self.housing)
        pred = model(self.housing.X)
        self.assertEqual(len(self.housing), len(pred))
        self.assertGreater(all(pred), 0)

    @test_learners
    def test_predict_sparse(self, learner_class: XGBBase):
        sparse_data = self.housing.to_sparse()
        booster = learner_class()
        model = booster(sparse_data)
        pred = model(sparse_data)
        self.assertEqual(len(sparse_data), len(pred))
        self.assertGreater(all(pred), 0)

    @test_learners
    def test_set_params(self, learner_class: XGBBase):
        booster = learner_class(n_estimators=42, max_depth=4)
        self.assertEqual(booster.params["n_estimators"], 42)
        self.assertEqual(booster.params["max_depth"], 4)
        model = booster(self.housing)
        params = model.skl_model.get_params()
        self.assertEqual(params["n_estimators"], 42)
        self.assertEqual(params["max_depth"], 4)

    @unittest.skipIf(XGBRegressor is None, "Missing 'xgboost' package")
    @test_learners
    def test_scorer(self, learner_class: XGBBase):
        booster = learner_class()
        self.assertIsInstance(booster, Scorer)
        booster.score(self.housing)


if __name__ == "__main__":
    unittest.main()
