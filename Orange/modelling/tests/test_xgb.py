import unittest
from typing import Callable, Union

from Orange.data import Table
from Orange.evaluation import CrossValidation

try:
    from Orange.modelling import XGBLearner, XGBRFLearner
except ImportError:
    XGBLearner = XGBRFLearner = None


def test_learners(func: Callable) -> Callable:
    def wrapper(self):
        func(self, XGBLearner)
        func(self, XGBRFLearner)

    return wrapper


@unittest.skipIf(XGBLearner is None, "Missing 'xgboost' package")
class TestXGB(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.iris = Table("iris")
        cls.housing = Table("housing")

    @test_learners
    def test_cls(self, learner_class: Union[XGBLearner, XGBRFLearner]):
        booster = learner_class()
        cv = CrossValidation(k=10)
        cv(self.iris, [booster])

    @test_learners
    def test_reg(self, learner_class: Union[XGBLearner, XGBRFLearner]):
        booster = learner_class()
        cv = CrossValidation(k=10)
        cv(self.housing, [booster])

    @test_learners
    def test_params(self, learner_class: Union[XGBLearner, XGBRFLearner]):
        booster = learner_class(n_estimators=42, max_depth=4)
        self.assertEqual(booster.get_params(self.iris)["n_estimators"], 42)
        self.assertEqual(booster.get_params(self.housing)["n_estimators"], 42)
        self.assertEqual(booster.get_params(self.iris)["max_depth"], 4)
        self.assertEqual(booster.get_params(self.housing)["max_depth"], 4)
        model = booster(self.housing)
        params = model.skl_model.get_params()
        self.assertEqual(params["n_estimators"], 42)
        self.assertEqual(params["max_depth"], 4)

    @test_learners
    def test_scorer(self, learner_class: Union[XGBLearner, XGBRFLearner]):
        booster = learner_class()
        booster.score(self.iris)
        booster.score(self.housing)


if __name__ == "__main__":
    unittest.main()
