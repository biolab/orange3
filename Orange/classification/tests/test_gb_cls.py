import unittest

from Orange.classification import GBClassifier
from Orange.data import Table
from Orange.evaluation import CrossValidation, CA
from Orange.preprocess.score import Scorer


class TestGBClassifier(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.iris = Table("iris")

    def test_GBTrees(self):
        booster = GBClassifier()
        cv = CrossValidation(k=10)
        results = cv(self.iris, [booster])
        ca = CA(results)
        self.assertGreater(ca, 0.9)
        self.assertLess(ca, 0.99)

    def test_predict_single_instance(self):
        booster = GBClassifier()
        model = booster(self.iris)
        for ins in self.iris:
            model(ins)
            prob = model(ins, model.Probs)
            self.assertGreaterEqual(prob.all(), 0)
            self.assertLessEqual(prob.all(), 1)
            self.assertAlmostEqual(prob.sum(), 1, 3)

    def test_predict_table(self):
        booster = GBClassifier()
        model = booster(self.iris)
        pred = model(self.iris)
        self.assertEqual(pred.shape, (len(self.iris),))
        prob = model(self.iris, model.Probs)
        self.assertGreaterEqual(prob.all().all(), 0)
        self.assertLessEqual(prob.all().all(), 1)
        self.assertAlmostEqual(prob.sum(), len(self.iris))

    def test_predict_numpy(self):
        booster = GBClassifier()
        model = booster(self.iris)
        pred = model(self.iris.X)
        self.assertEqual(pred.shape, (len(self.iris),))
        prob = model(self.iris.X, model.Probs)
        self.assertGreaterEqual(prob.all().all(), 0)
        self.assertLessEqual(prob.all().all(), 1)
        self.assertAlmostEqual(prob.sum(), len(self.iris))

    def test_predict_sparse(self):
        sparse_data = self.iris.to_sparse()
        booster = GBClassifier()
        model = booster(sparse_data)
        pred = model(sparse_data)
        self.assertEqual(pred.shape, (len(sparse_data),))
        prob = model(sparse_data, model.Probs)
        self.assertGreaterEqual(prob.all().all(), 0)
        self.assertLessEqual(prob.all().all(), 1)
        self.assertAlmostEqual(prob.sum(), len(sparse_data))

    def test_default_params(self):
        booster = GBClassifier()
        model = booster(self.iris)
        self.assertDictEqual(booster.params, model.skl_model.get_params())

    def test_set_params(self):
        booster = GBClassifier(n_estimators=42, max_depth=4)
        self.assertEqual(booster.params["n_estimators"], 42)
        self.assertEqual(booster.params["max_depth"], 4)
        model = booster(self.iris)
        params = model.skl_model.get_params()
        self.assertEqual(params["n_estimators"], 42)
        self.assertEqual(params["max_depth"], 4)

    def test_scorer(self):
        booster = GBClassifier()
        self.assertIsInstance(booster, Scorer)
        booster.score(self.iris)


if __name__ == "__main__":
    unittest.main()
