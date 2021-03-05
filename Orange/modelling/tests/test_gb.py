import unittest

from Orange.data import Table
from Orange.evaluation import CrossValidation
from Orange.modelling import GBLearner


class TestGBLearner(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.iris = Table("iris")
        cls.housing = Table("housing")

    def test_cls(self):
        booster = GBLearner()
        cv = CrossValidation(k=10)
        cv(self.iris, [booster])

    def test_reg(self):
        booster = GBLearner()
        cv = CrossValidation(k=10)
        cv(self.housing, [booster])

    def test_params(self):
        booster = GBLearner(n_estimators=42, max_depth=4)
        self.assertEqual(booster.get_params(self.iris)["n_estimators"], 42)
        self.assertEqual(booster.get_params(self.housing)["n_estimators"], 42)
        self.assertEqual(booster.get_params(self.iris)["max_depth"], 4)
        self.assertEqual(booster.get_params(self.housing)["max_depth"], 4)
        model = booster(self.housing)
        params = model.skl_model.get_params()
        self.assertEqual(params["n_estimators"], 42)
        self.assertEqual(params["max_depth"], 4)

    def test_scorer(self):
        booster = GBLearner()
        booster.score(self.iris)
        booster.score(self.housing)


if __name__ == "__main__":
    unittest.main()
