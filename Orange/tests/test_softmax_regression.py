import unittest

from Orange.data import ContinuousVariable, Domain, Table
from Orange.classification import Model, SoftmaxRegressionLearner
from Orange.evaluation import CrossValidation, CA
import numpy as np


class SoftmaxRegressionTest(unittest.TestCase):
    def test_SoftmaxRegression(self):
        table = Table('iris')
        learner = SoftmaxRegressionLearner()
        results = CrossValidation(table, [learner], k=3)
        ca = CA(results)
        self.assertTrue(0.9 < ca < 1.0)

    def test_SoftmaxRegressionPreprocessors(self):
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
        learners = [SoftmaxRegressionLearner(preprocessors=[]),
                    SoftmaxRegressionLearner()]
        results = CrossValidation(table, learners, k=3)
        ca = CA(results)
        self.assertTrue(ca[0] < ca[1])

    def test_probability(self):
        table = Table('iris')
        learn = SoftmaxRegressionLearner()
        clf = learn(table)
        p = clf(table, ret=Model.Probs)
        self.assertTrue(all(abs(p.sum(axis=1) - 1) < 1e-6))

    def test_predict_table(self):
        table = Table('iris')
        learner = SoftmaxRegressionLearner()
        c = learner(table)
        c(table)
        vals, probs = c(table, c.ValueProbs)

    def test_predict_numpy(self):
        table = Table('iris')
        learner = SoftmaxRegressionLearner()
        c = learner(table)
        c(table.X)
        vals, probs = c(table.X, c.ValueProbs)
