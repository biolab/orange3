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

    def test_LogisticRegressionPreprocessors(self):
        np.random.seed(42)
        table = Table('iris')
        new_attrs = (ContinuousVariable('c0'),) + table.domain.attributes
        new_domain = Domain(new_attrs,
                            table.domain.class_vars,
                            table.domain.metas)
        new_table = np.hstack((
            1000000 * np.random.random((table.X.shape[0], 1)),
            table))
        table = table.from_numpy(new_domain, new_table)
        learners = [LogisticRegressionLearner(preprocessors=[]),
                    LogisticRegressionLearner()]
        results = CrossValidation(table, learners, k=3)
        ca = CA(results)
        self.assertTrue(ca[0] < ca[1])

    def test_probability(self):
        table = Table('iris')
        learn = LogisticRegressionLearner(penalty='l1')
        clf = learn(table[:100])
        p = clf(table[100:], ret=Model.Probs)
        self.assertTrue(all(abs(p.sum(axis=1) - 1) < 1e-6))
