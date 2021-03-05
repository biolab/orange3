import unittest
from typing import Callable

from Orange.base import XGBBase

try:
    from Orange.classification import XGBClassifier, XGBRFClassifier
except ImportError:
    XGBClassifier = XGBRFClassifier = None
from Orange.data import Table
from Orange.evaluation import CrossValidation, CA
from Orange.preprocess.score import Scorer


def test_learners(func: Callable) -> Callable:
    def wrapper(self):
        func(self, XGBClassifier)
        func(self, XGBRFClassifier)

    return wrapper


@unittest.skipIf(XGBClassifier is None, "Missing 'xgboost' package")
class TestXGBCls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.iris = Table("iris")

    @test_learners
    def test_XGB(self, learner_class: XGBBase):
        booster = learner_class()
        cv = CrossValidation(k=10)
        results = cv(self.iris, [booster])
        ca = CA(results)
        self.assertGreater(ca, 0.9)
        self.assertLess(ca, 0.99)

    @test_learners
    def test_predict_single_instance(self, learner_class: XGBBase):
        booster = learner_class()
        model = booster(self.iris)
        for ins in self.iris:
            model(ins)
            prob = model(ins, model.Probs)
            self.assertGreaterEqual(prob.all(), 0)
            self.assertLessEqual(prob.all(), 1)
            self.assertAlmostEqual(prob.sum(), 1, 3)

    @test_learners
    def test_predict_table(self, learner_class: XGBBase):
        booster = learner_class()
        model = booster(self.iris)
        pred = model(self.iris)
        self.assertEqual(len(self.iris), len(pred))
        prob = model(self.iris, model.Probs)
        self.assertGreaterEqual(prob.all().all(), 0)
        self.assertLessEqual(prob.all().all(), 1)
        self.assertAlmostEqual(prob.sum(), len(self.iris))

    @test_learners
    def test_predict_numpy(self, learner_class: XGBBase):
        booster = learner_class()
        model = booster(self.iris)
        pred = model(self.iris.X)
        self.assertEqual(len(self.iris), len(pred))
        prob = model(self.iris.X, model.Probs)
        self.assertGreaterEqual(prob.all().all(), 0)
        self.assertLessEqual(prob.all().all(), 1)
        self.assertAlmostEqual(prob.sum(), len(self.iris))

    @test_learners
    def test_predict_sparse(self, learner_class: XGBBase):
        sparse_data = self.iris.to_sparse()
        booster = learner_class()
        model = booster(sparse_data)
        pred = model(sparse_data)
        self.assertEqual(len(sparse_data), len(pred))
        prob = model(sparse_data, model.Probs)
        self.assertGreaterEqual(prob.all().all(), 0)
        self.assertLessEqual(prob.all().all(), 1)
        self.assertAlmostEqual(prob.sum(), len(sparse_data))

    @test_learners
    def test_set_params(self, learner_class: XGBBase):
        booster = learner_class(n_estimators=42, max_depth=4)
        self.assertEqual(booster.params["n_estimators"], 42)
        self.assertEqual(booster.params["max_depth"], 4)
        model = booster(self.iris)
        params = model.skl_model.get_params()
        self.assertEqual(params["n_estimators"], 42)
        self.assertEqual(params["max_depth"], 4)

    @test_learners
    def test_scorer(self, learner_class: XGBBase):
        booster = learner_class()
        self.assertIsInstance(booster, Scorer)
        booster.score(self.iris)


if __name__ == "__main__":
    unittest.main()
