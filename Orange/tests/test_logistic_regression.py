import unittest

from Orange.data import Table, ContinuousVariable, Domain
from Orange.classification import LogisticRegressionLearner, Model
from Orange.evaluation import CrossValidation, CA
import numpy as np


class LogisticRegressionTest(unittest.TestCase):
    def test_LogisticRegression(self):
        table = Table('iris')
        learn = LogisticRegressionLearner()
        results = CrossValidation(table, [learn], k=2)
        ca = CA(results)
        self.assertTrue(0.8 < ca < 1.0)

    def test_probability(self):
        table = Table('iris')
        learn = LogisticRegressionLearner(penalty='l1')
        clf = learn(table[:100])
        p = clf(table[100:], ret=Model.Probs)
        self.assertTrue(all(abs(p.sum(axis=1) - 1) < 1e-6))
