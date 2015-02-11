import unittest

import Orange
from Orange.classification import LogisticRegressionLearner

class LogisticRegressionTest(unittest.TestCase):

    def test_LogisticRegression(self):
        table = Orange.data.Table('iris')
        learn = LogisticRegressionLearner()
        results = Orange.evaluation.CrossValidation(table, [learn], k=2)
        ca = Orange.evaluation.CA(results)
        self.assertTrue(0.9 < ca < 1.0)

    def test_probability(self):
        table = Orange.data.Table('iris')
        learn = LogisticRegressionLearner(penalty='l1')
        clf = learn(table[:100])
        p = clf(table[100:], ret=Orange.classification.Model.Probs)
        self.assertTrue(all(abs(p.sum(axis=1)-1) < 1e-6))
