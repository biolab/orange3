import unittest

import Orange
from Orange.classification import SoftmaxRegressionLearner


class SoftmaxRegressionTest(unittest.TestCase):
    def test_SoftmaxRegression(self):
        table = Orange.data.Table('iris')
        learner = SoftmaxRegressionLearner()
        results = Orange.evaluation.CrossValidation(table, [learner], k=3)
        ca = Orange.evaluation.CA(results)
        self.assertTrue(0.9 < ca < 1.0)

    def test_probability(self):
        table = Orange.data.Table('iris')
        learn = SoftmaxRegressionLearner()
        clf = learn(table)
        p = clf(table, ret=Orange.classification.Model.Probs)
        self.assertTrue(all(abs(p.sum(axis=1) - 1) < 1e-6))

    def test_predict_table(self):
        table = Orange.data.Table('iris')
        learner = SoftmaxRegressionLearner()
        c = learner(table)
        c(table)
        vals, probs = c(table, c.ValueProbs)

    def test_predict_numpy(self):
        table = Orange.data.Table('iris')
        learner = SoftmaxRegressionLearner()
        c = learner(table)
        c(table.X)
        vals, probs = c(table.X, c.ValueProbs)
