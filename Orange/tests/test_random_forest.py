import unittest

import Orange.data
import Orange.classification.random_forest as rf
from Orange.evaluation import scoring, testing


class RandomForestTest(unittest.TestCase):
    def test_RandomForest(self):
        table = Orange.data.Table('titanic')
        forest = rf.RandomForestLearner()
        results = testing.CrossValidation(table[::20], [forest], k=10)
        ca = scoring.CA(results)
        self.assertGreater(ca, 0.7)
        self.assertLess(ca, 0.9)

    def test_predict_single_instance(self):
        table = Orange.data.Table('titanic')
        forest = rf.RandomForestLearner()
        c = forest(table)
        for ins in table[::20]:
            c(ins)
            val, prob = c(ins, c.ValueProbs)

    def test_predict_table(self):
        table = Orange.data.Table('titanic')
        forest = rf.RandomForestLearner()
        c = forest(table)
        table = table[::20]
        c(table)
        vals, probs = c(table, c.ValueProbs)

    def test_predict_numpy(self):
        table = Orange.data.Table('titanic')
        forest = rf.RandomForestLearner()
        c = forest(table)
        X = table.X[::20]
        c(X)
        vals, probs = c(X, c.ValueProbs)