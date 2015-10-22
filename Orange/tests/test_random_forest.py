import unittest
import numpy as np
from Orange.data import Table
from Orange.evaluation import CrossValidation, CA, RMSE
from Orange.classification import RandomForestLearner
from Orange.regression import RandomForestRegressionLearner


class RandomForestTest(unittest.TestCase):
    def test_RandomForest(self):
        table = Table('iris')
        forest = RandomForestLearner()
        results = CrossValidation(table, [forest], k=10)
        ca = CA(results)
        self.assertGreater(ca, 0.9)
        self.assertLess(ca, 0.99)

    def test_predict_single_instance(self):
        table = Table('iris')
        forest = RandomForestLearner()
        c = forest(table)
        for ins in table:
            c(ins)
            val, prob = c(ins, c.ValueProbs)

    def test_predict_table(self):
        table = Table('iris')
        forest = RandomForestLearner()
        c = forest(table)
        c(table)
        vals, probs = c(table, c.ValueProbs)

    def test_predict_numpy(self):
        table = Table('iris')
        forest = RandomForestLearner()
        c = forest(table)
        c(table.X)
        vals, probs = c(table.X, c.ValueProbs)

    def test_RandomForestRegression(self):
        table = Table('housing')
        forest = RandomForestRegressionLearner()
        results = CrossValidation(table, [forest], k=10)
        _ = RMSE(results)

    def test_predict_single_instance_reg(self):
        table = Table('housing')
        forest = RandomForestRegressionLearner()
        model = forest(table)
        for ins in table:
            pred = model(ins)
            self.assertTrue(pred > 0)

    def test_predict_table_reg(self):
        table = Table('housing')
        forest = RandomForestRegressionLearner()
        model = forest(table)
        pred = model(table)
        self.assertEqual(len(table), len(pred))
        self.assertTrue(all(pred) > 0)

    def test_predict_numpy_reg(self):
        table = Table('housing')
        forest = RandomForestRegressionLearner()
        model = forest(table)
        pred = model(table.X)
        self.assertEqual(len(table), len(pred))
        self.assertTrue(all(pred) > 0)

    def test_scorer(self):
        data = Table('test4.tab')
        learner = RandomForestLearner()
        scores = learner.score_data(data)
        self.assertEqual(len(scores), len(data.domain.attributes))
        self.assertNotEqual(sum(scores), 0)

    def test_scorer_feature(self):
        np.random.seed(42)
        data = Table('test4.tab')
        learner = RandomForestLearner()
        scores = learner.score_data(data)
        for i, attr in enumerate(data.domain.attributes):
            np.random.seed(42)
            score = learner.score_data(data, attr)
            self.assertEqual(score, scores[i])
