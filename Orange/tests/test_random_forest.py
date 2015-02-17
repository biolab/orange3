import unittest

import Orange
from Orange.classification import RandomForestLearner


class RandomForestTest(unittest.TestCase):
    def test_RandomForest(self):
        table = Orange.data.Table('iris')
        forest = RandomForestLearner()
        results = Orange.evaluation.CrossValidation(table, [forest], k=10)
        ca = Orange.evaluation.CA(results)
        self.assertGreater(ca, 0.9)
        self.assertLess(ca, 0.99)

    def test_predict_single_instance(self):
        table = Orange.data.Table('iris')
        forest = RandomForestLearner()
        c = forest(table)
        for ins in table:
            c(ins)
            val, prob = c(ins, c.ValueProbs)

    def test_predict_table(self):
        table = Orange.data.Table('iris')
        forest = RandomForestLearner()
        c = forest(table)
        c(table)
        vals, probs = c(table, c.ValueProbs)

    def test_predict_numpy(self):
        table = Orange.data.Table('iris')
        forest = RandomForestLearner()
        c = forest(table)
        c(table.X)
        vals, probs = c(table.X, c.ValueProbs)
