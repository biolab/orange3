import sys
import unittest
from unittest.mock import patch
from typing import Callable, Union

from Orange.data import Table
from Orange.evaluation import CrossValidation
from Orange.modelling import XGBLearner, XGBRFLearner


def test_learners(func: Callable) -> Callable:
    def wrapper(self):
        func(self, XGBLearner)
        func(self, XGBRFLearner)

    return wrapper


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

    def test_import_missing_library(self):
        modules = {k: v for k, v in sys.modules.items()
                   if "orange" not in k.lower()}  # retain built-ins
        modules["xgboost"] = None
        # pylint: disable=reimported,redefined-outer-name
        # pylint: disable=unused-import,import-outside-toplevel
        with patch.dict(sys.modules, modules, clear=True):
            def import_():
                from Orange.modelling import XGBLearner

            self.assertRaises(ImportError, import_)


if __name__ == "__main__":
    unittest.main()
